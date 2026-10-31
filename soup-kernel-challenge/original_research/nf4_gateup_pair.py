"""Original paired NF4 decoder for frozen Qwen MLP gate/up projections.

Independent original implementation derived from the existing own solution/nf4.py
and NF4's documented quantization contract. This does NOT import or copy either
competitor's kernels. Only merges the two data-parallel decode calls into one
2-dimensional Triton launch, preserving separate quantization metadata.
"""

import contextvars
import atexit
import json
from functools import wraps
from importlib.metadata import version
import math

import torch
import triton
import triton.language as tl

from . import nf4 as _base_nf4

_ACTIVE = contextvars.ContextVar("_soup_nf4_gate_up_group", default=None)
_GROUPS = {}
_BY_PACKED_ADDRESS = {}
_COUNTS = {"compiled_groups": 0, "forward_pair_decodes": 0,
           "backward_pair_decodes": 0, "forward_matmuls": 0,
           "backward_reuses": 0}


@triton.jit(do_not_specialize=["n_elements"])
def soup_nf4_paired_gate_up(
    packed_g, scales_g, nested_g, table_g, offset_g, output_g,
    packed_u, scales_u, nested_u, table_u, offset_u, output_u,
    levels, n_elements,
    QUANT_BLOCK: tl.constexpr, SCALE_BLOCK: tl.constexpr, CHUNK: tl.constexpr,
):
    """Two independent NF4 matrices, each with its original code/offset tables.

    The second launch dimension selects a weight, not a fake extra computation.
    One byte holds two consecutive NF4 indices; packed 32-bit stores write the
    two FP16 values without transposition or a second output tensor.
    """
    which = tl.program_id(1)
    flat_byte = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    mask = flat_byte * 2 < n_elements

    src = tl.where(which == 0, packed_g, packed_u)
    scale_ptr = tl.where(which == 0, scales_g, scales_u)
    nested_ptr = tl.where(which == 0, nested_g, nested_u)
    code_ptr = tl.where(which == 0, table_g, table_u)
    zero_ptr = tl.where(which == 0, offset_g, offset_u)
    dst = tl.where(which == 0, output_g, output_u)

    nibble_pair = tl.load(src + flat_byte, mask, 0).to(tl.int32)
    scale_index = (flat_byte * 2) // QUANT_BLOCK
    nested_index = scale_index // SCALE_BLOCK
    code_index = tl.load(scale_ptr + scale_index, mask, 0).to(tl.int32)
    outer_code = tl.load(code_ptr + code_index)
    outer_scale = tl.load(nested_ptr + nested_index, mask, 0)
    outer_offset = tl.load(zero_ptr)
    # FMA must be disabled at launch to match bitsandbytes' separate arithmetic.
    final_scale = outer_code * outer_scale + outer_offset
    hi = tl.load(levels + (nibble_pair >> 4)) * final_scale
    lo = tl.load(levels + (nibble_pair & 15)) * final_scale
    hi16 = hi.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    lo16 = lo.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    words = hi16 | (lo16 << 16)
    tl.store(dst.to(tl.pointer_type(tl.uint32)) + flat_byte,
             words, mask & (2 * flat_byte + 1 < n_elements))
    tl.store(dst + 2 * flat_byte, hi, mask & (2 * flat_byte == n_elements - 1))


class _GateUpPair:
    def __init__(self, mlp):
        g, u = mlp.gate_proj, mlp.up_proj
        self.packed = (g.weight, u.weight)
        self.states = (g.weight.quant_state, u.weight.quant_state)
        comps = [_base_nf4._state_components(w, s, None, cuda_only=True)
                 for w, s in zip(self.packed, self.states)]
        if (any(c is None for c in comps) or
            any(c["dtype"] != torch.float16 or c["nested_scales"] is None
                for c in comps) or
            tuple(comps[0]["shape"]) != tuple(comps[1]["shape"]) or
            comps[0]["blocksize"] != comps[1]["blocksize"] or
            comps[0]["nested_blocksize"] != comps[1]["nested_blocksize"] or
            any(c["packed"].data_ptr() == 0 for c in comps)):
            raise ValueError("not a supported pair of frozen, nested NF4 weights")
        self.components = comps
        self.shape = tuple(comps[0]["shape"])
        self.n_elements = math.prod(self.shape)
        self.forward_saved = None
        self.backward_saved = None
        self.backward_left = 0
        self.addresses = {w.data_ptr(): i for i, w in enumerate(self.packed)}

    def decode(self, phase):
        g, u = self.components
        output = (torch.empty(self.shape, dtype=torch.float16, device=g["packed"].device),
                  torch.empty(self.shape, dtype=torch.float16, device=u["packed"].device))
        assert all(t.data_ptr() % 4 == 0 for t in output)
        levels = _base_nf4._nf4_table(g["packed"].device)
        soup_nf4_paired_gate_up[(triton.cdiv(self.n_elements, 2048), 2)](
            g["packed"], g["scales"], g["nested_scales"], g["nested_code"], g["offset"], output[0],
            u["packed"], u["scales"], u["nested_scales"], u["nested_code"], u["offset"], output[1],
            levels, self.n_elements,
            QUANT_BLOCK=g["blocksize"], SCALE_BLOCK=g["nested_blocksize"],
            CHUNK=1024, num_warps=4, enable_fp_fusion=False,
        )
        _COUNTS[phase + "_pair_decodes"] += 1
        return output

    def forward_weight(self, address):
        if self.forward_saved is None:
            self.forward_saved = self.decode("forward")
        _COUNTS["forward_matmuls"] += 1
        return self.forward_saved[self.addresses[address]]

    def backward_weight(self, address):
        if self.backward_saved is None:
            self.backward_saved = self.decode("backward")
            self.backward_left = 2
        found = self.backward_saved[self.addresses[address]]
        _COUNTS["backward_reuses"] += 1
        self.backward_left -= 1
        if self.backward_left <= 0:
            self.backward_saved = None
        return found


def patch_mlp_pair_decoder():
    """Guarded installation; on any unsupported model keep the base NF4 path."""
    if not torch.cuda.is_available() or version("bitsandbytes") != "0.50.2":
        return False
    import bitsandbytes.functional as functional
    from bitsandbytes.backends.cuda import ops
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    if getattr(Qwen2DecoderLayer.forward, "_soup_nf4_gateup", False):
        return True
    orig_decoder = Qwen2DecoderLayer.forward
    orig_dequant = functional.dequantize_4bit
    orig_linear = ops._dequant_linear_fallback
    if not getattr(orig_dequant, "_soup_candidate_nf4", False):
        return False
    if not getattr(orig_linear, "_soup_candidate_nf4", False):
        return False

    @wraps(orig_decoder)
    def wrapped_decoder(self, *args, **kwargs):
        pair = _GROUPS.get(id(self))
        if pair is None and self.training and torch.is_grad_enabled():
            try:
                pair = _GateUpPair(self.mlp)
            except (AttributeError, ValueError, TypeError):
                pair = False
            _GROUPS[id(self)] = pair
            if pair:
                _BY_PACKED_ADDRESS.update((key, pair) for key in pair.addresses)
                _COUNTS["compiled_groups"] += 1
        if not pair or not self.training or not torch.is_grad_enabled():
            return orig_decoder(self, *args, **kwargs)
        pair.forward_saved = None
        token = _ACTIVE.set(pair)
        try:
            return orig_decoder(self, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)
            pair.forward_saved = None

    @wraps(orig_linear)
    def grouped_linear(A, B, shapeB, absmax, blocksize, quant_type,
                       bias=None, absmax_8bit=None, absmax_code=None,
                       absmax_offset=None):
        pair = _ACTIVE.get()
        if (pair and A.dtype == torch.float16 and quant_type == "nf4" and
            B.data_ptr() in pair.addresses and B.is_cuda and
            tuple(shapeB) == pair.shape):
            decoded = pair.forward_weight(B.data_ptr())
            return torch.nn.functional.linear(A, decoded, bias)
        return orig_linear(A, B, shapeB, absmax, blocksize, quant_type, bias,
                           absmax_8bit, absmax_code, absmax_offset)

    @wraps(orig_dequant)
    def grouped_dequant(A, quant_state=None, absmax=None, out=None,
                        blocksize=None, quant_type="fp4"):
        pair = _BY_PACKED_ADDRESS.get(A.data_ptr()) if A.is_cuda else None
        if (pair and out is None and any(quant_state is t for t in pair.states) and
            quant_type in ("nf4", "fp4") and A.dtype == torch.uint8):
            decoded = pair.backward_weight(A.data_ptr())
            return decoded.t() if A.shape[0] == 1 else decoded
        return orig_dequant(A, quant_state, absmax, out, blocksize, quant_type)

    wrapped_decoder._soup_nf4_gateup = True
    grouped_linear._soup_candidate_nf4 = True
    grouped_dequant._soup_candidate_nf4 = True
    Qwen2DecoderLayer.forward = wrapped_decoder
    ops._dequant_linear_fallback = grouped_linear
    functional.dequantize_4bit = grouped_dequant
    return True


def _report():
    if _COUNTS["forward_pair_decodes"] or _COUNTS["backward_pair_decodes"]:
        print("[soup_original_nf4_pair] " + json.dumps(_COUNTS, sort_keys=True), flush=True)


atexit.register(_report)