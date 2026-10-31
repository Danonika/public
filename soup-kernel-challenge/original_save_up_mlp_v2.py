"""Original save-up-only frozen Qwen2 MLP activation optimization.

Idea-level inspiration: public Soup competitors' saved-activation studies.
Implementation is independent. Existing unchanged starter SwiGLU kernels are
used for forward/backward; starter kernels are NOT countable as own kernels.

Only the task's frozen, NF4 4-bit MLP is supported. Save only FP16 up and the
already-live FP32 MLP input; recompute gate once in backward. No MLP LoRA,
no dropout and no RNG usage in this supported path. The ordinary gate/up
projection graphs still contribute their independently rounded FP16 gradients.

Install before the Qwen2 MLP padding wrapper so it captures this new MLP path.
"""

from functools import wraps

import torch
import triton

from .swiglu import BLOCK, example_swiglu_fwd_kernel, example_swiglu_bwd_kernel
from .activation_storage import _eligible_mlp, compact_swiglu


class _SaveUpSwiGLU(torch.autograd.Function):
    """Save up and x, not gate; leave the two projection graphs unchanged."""

    @staticmethod
    def forward(ctx, gate_half, up_half, gate_input_half, frozen_gate_proj):
        if gate_half.dtype != torch.float16 or up_half.dtype != torch.float16:
            raise TypeError("Save-up requires two FP16 projection outputs")
        if gate_half.shape != up_half.shape or not gate_half.is_contiguous() or not up_half.is_contiguous():
            raise ValueError("Save-up requires matching contiguous activations")
        if gate_input_half.dtype != torch.float16 or not gate_input_half.is_contiguous():
            raise ValueError("Save-up requires exact FP16 gate projection input")
        out = torch.empty_like(gate_half, dtype=torch.float32)
        n = gate_half.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate_half, up_half, out, n, BLOCK=BLOCK
        )
        ctx.save_for_backward(up_half, gate_input_half)
        ctx.gate_proj = frozen_gate_proj
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None, None
        up_half, gate_input_half = ctx.saved_tensors
        # Same Linear4bit autograd dispatch as original forward. This isolated
        # temporary graph is discarded; we need only gate's FP16 value.
        with torch.enable_grad():
            fresh_input = gate_input_half.detach().requires_grad_(True)
            gate_rebuilt = ctx.gate_proj(fresh_input)
        if (gate_rebuilt.dtype != up_half.dtype or
                gate_rebuilt.shape != up_half.shape or
                not gate_rebuilt.is_contiguous()):
            raise RuntimeError("Frozen gate projection changed shape/dtype in backward")
        dgate = torch.empty_like(gate_rebuilt)
        dup = torch.empty_like(up_half)
        grad = grad_out.to(torch.float32).contiguous()
        example_swiglu_bwd_kernel[(triton.cdiv(ctx.n, BLOCK),)](
            grad, gate_rebuilt, up_half, dgate, dup, ctx.n, BLOCK=BLOCK
        )
        # The original gate/up projection graphs supply dX. No additional
        # input gradient is applied to x_fp32 by this custom SwiGLU node.
        return dgate, dup, None, None


def patch_save_up_mlp():
    """Wrap only our audited frozen compact-activation MLP, idempotently."""
    import bitsandbytes as bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    previous = Qwen2MLP.forward
    if getattr(previous, "_soup_original_save_up", False):
        return True
    if not getattr(previous, "_soup_candidate_compact_activations", False):
        return False
    linear_cls = bnb.nn.Linear4bit
    linear_forward = linear_cls.forward

    @wraps(previous)
    def forward(self, x):
        if not _eligible_mlp(self, x, linear_cls, linear_forward):
            return previous(self, x)
        # Keep casts separate so gate/up dX branches round individually.
        gate_input_half = x.to(torch.float16)
        gate = self.gate_proj(gate_input_half)
        up = self.up_proj(x.to(torch.float16))
        if (gate.dtype != torch.float16 or up.dtype != torch.float16 or
                gate.shape != up.shape or not gate.is_contiguous() or
                not up.is_contiguous()):
            return self.down_proj(compact_swiglu(gate, up))
        return self.down_proj(_SaveUpSwiGLU.apply(gate, up, gate_input_half, self.gate_proj))

    forward._soup_original_save_up = True
    Qwen2MLP.forward = forward
    return True
