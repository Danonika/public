"""Adaptive448 + backward GEMM M-row padding (research-only).

Independently authored extension on pinned, ATTRIBUTED external contributor control.
Reuses our previously authored input-only MLP *forward* byte-for-byte by
subclassing its autograd.Function; only the >448, non-multiple-of-8 backward
row geometry is changed. This source is not a jury-original solution.
"""
from __future__ import annotations

import atexit
import torch
import torch.nn.functional as F
import triton
from .own_input_only_grouped_mlp import _RecomputeBoth
from .mlp_save_up import save_up_group
from .state import DISABLED
from .swiglu import BLOCK, example_swiglu_bwd_kernel

_GATE, _UP = 4, 5
_STEP = 8
_STATS = {"padded": 0, "bwd": 0, "short_native": 0,
          "aligned_parent": 0, "fallback": 0, "extra_rows": 0}


class _PadBackwardOnly(_RecomputeBoth):
    """Forward is EXACTLY the parent; backward extends M rows to 8-alignment."""
    @staticmethod
    def backward(ctx, dy):
        if dy is None:
            return None, None
        (x,) = ctx.saved_tensors
        group = ctx.group
        xshape = tuple(x.shape)
        rows = x.numel() // xshape[-1]
        extra = (-rows) % _STEP
        if rows <= 448 or extra == 0:
            raise RuntimeError("Row-padding branch used on a noneligible shape")
        xpad = F.pad(x.reshape(rows, xshape[-1]), (0, 0, 0, extra))
        # Keep original separate FP16 GEMMs and two independent FP16-roundings.
        wg = group.weight("bwd", _GATE)
        wu = group.weight("bwd", _UP)
        g = F.linear(xpad, wg, group.biases[_GATE])
        u = F.linear(xpad, wu, group.biases[_UP])
        dyp = F.pad(dy.reshape(rows, u.shape[-1]).to(g.dtype).contiguous(),
                    (0, 0, 0, extra))
        count = g.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(count, BLOCK),)](
            dyp, g, u, g, u, count, BLOCK=BLOCK)
        del dyp
        dx = torch.matmul(g, wg)
        dx.add_(torch.matmul(u, wu))
        _STATS["bwd"] += 1
        return dx[:rows].reshape(xshape), None


def install_adaptive448_rowpad():
    from transformers.models.qwen2 import modeling_qwen2
    cls = modeling_qwen2.Qwen2MLP
    prior = cls.forward
    if getattr(prior, "_adaptive448_rowpad", False):
        return True
    if not getattr(prior, "_soup_input_only_replay", False):
        raise RuntimeError("Expected exact measured adaptive448 input-only parent")

    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or not x.is_cuda
                or x.dtype != torch.float16 or x.ndim not in (2, 3)
                or x.shape[-1] != 1536 or not x.is_contiguous()
                or not x.requires_grad or not torch.is_grad_enabled()
                or not self.training or "qgroup" in DISABLED):
            _STATS["fallback"] += 1
            return prior(self, x)
        rows = x.numel() // 1536
        if rows <= 448:
            _STATS["short_native"] += 1
            return prior(self, x)
        if rows % _STEP == 0:
            _STATS["aligned_parent"] += 1
            return prior(self, x)
        group = save_up_group(self, x)
        if (group is None or group.checkpointed
                or tuple(group.shapes[_GATE]) != (8960, 1536)
                or tuple(group.shapes[_UP]) != (8960, 1536)):
            _STATS["fallback"] += 1
            return prior(self, x)
        activation = _PadBackwardOnly.apply(x, group)
        group.expected_bwd += 2
        _STATS["padded"] += 1
        _STATS["extra_rows"] += (-rows) % _STEP
        return self.down_proj(activation)

    forward._adaptive448_rowpad = True
    forward._previous = prior
    cls.forward = forward
    atexit.register(lambda: print("[own-adaptive448-rowpad] " + repr(_STATS)))
    return True
