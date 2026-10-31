"""Independently written grouped-NF4-aware, input-only frozen MLP autograd.

This computes the SAME mathematical SiLU(gate) * up as the starter example
in forward. Backward replays both native FP16 gate/up GEMMs using the frozen
group's existing backward dequant buffer, never saving the two wide activations.
Original research implementation; all other runtime source is attributed
external contributor code and cannot be claimed as ours for a contest submission.
"""
import atexit
import torch
import torch.nn.functional as F
import triton
from .mlp_save_up import save_up_group
from .swiglu import BLOCK, example_swiglu_fwd_kernel, example_swiglu_bwd_kernel

_STATE={"active":0,"fallback":0,"bw":0}
GATE,UP=4,5

class _RecomputeBoth(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, group):
        gate_w=group.weight("fwd",GATE)
        gate=F.linear(inputs,gate_w,group.biases[GATE])
        del gate_w
        up_w=group.weight("fwd",UP)
        up=F.linear(inputs,up_w,group.biases[UP])
        del up_w
        count=gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(count,BLOCK),)](
            gate,up,gate,count,BLOCK=BLOCK
        )
        ctx.group=group
        ctx.save_for_backward(inputs)
        return gate

    @staticmethod
    def backward(ctx,grad_output):
        (inputs,)=ctx.saved_tensors
        group=ctx.group
        gate_w=group.weight("bwd",GATE)
        up_w=group.weight("bwd",UP)
        gate=F.linear(inputs,gate_w,group.biases[GATE])
        up=F.linear(inputs,up_w,group.biases[UP])
        n=gate.numel()
        dy=grad_output.to(gate.dtype).contiguous()
        example_swiglu_bwd_kernel[(triton.cdiv(n,BLOCK),)](
            dy,gate,up,gate,up,n,BLOCK=BLOCK
        )
        dx=torch.matmul(gate,gate_w)
        dx.add_(torch.matmul(up,up_w))
        _STATE["bw"]+=1
        return dx,None

def install_input_only_replay():
    from transformers.models.qwen2 import modeling_qwen2
    from .state import DISABLED
    cls=modeling_qwen2.Qwen2MLP
    original=cls.forward
    if getattr(original,"_soup_input_only_replay",False):
        return True

    def forward(self,x):
        group = None
        if ("qgroup" not in DISABLED and x.is_cuda and
                torch.is_grad_enabled() and x.requires_grad):
            group = save_up_group(self,x)
        if group is None or group.checkpointed:
            _STATE["fallback"]+=1
            return original(self,x)
        activation=_RecomputeBoth.apply(x,group)
        group.expected_bwd+=2
        _STATE["active"]+=1
        return self.down_proj(activation)

    forward._soup_input_only_replay=True
    forward._previous=original
    cls.forward=forward
    atexit.register(lambda: print("[our-replay] "+repr(_STATE)))
    return True
