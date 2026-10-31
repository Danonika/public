"""Original one-node, input-only frozen MLP experiment.

Research-only overlay on ATTRIBUTED external contributor ce62d67 control and independent
input-only replay. Never credit competitor/starter kernels as our own.
"""
from __future__ import annotations
import atexit
import torch
import torch.nn.functional as F
import triton
from .mlp_save_up import save_up_group
from .state import DISABLED
from .swiglu import BLOCK, example_swiglu_fwd_kernel, example_swiglu_bwd_kernel

GATE,UP,DOWN=4,5,6
HIDDEN,INTERMEDIATE=1536,8960
_COUNTERS={"engaged":0,"fallback":0,"bwd":0}

class _OneNodeMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,group):
        wg=group.weight("fwd",GATE)
        gate=F.linear(x,wg,group.biases[GATE])
        del wg
        wu=group.weight("fwd",UP)
        up=F.linear(x,wu,group.biases[UP])
        del wu
        n=gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n,BLOCK),)](gate,up,gate,n,BLOCK=BLOCK)
        del up
        wd=group.weight("fwd",DOWN)
        y=F.linear(gate,wd,group.biases[DOWN])
        del gate,wd
        ctx.group=group
        ctx.save_for_backward(x)
        return y
    @staticmethod
    def backward(ctx,dy):
        if dy is None:
            return None,None
        (x,)=ctx.saved_tensors
        group=ctx.group
        wd=group.weight("bwd",DOWN)
        dact=torch.matmul(dy,wd)
        del wd
        wg=group.weight("bwd",GATE)
        wu=group.weight("bwd",UP)
        gate=F.linear(x,wg,group.biases[GATE])
        up=F.linear(x,wu,group.biases[UP])
        n=gate.numel()
        dact=dact.to(gate.dtype).contiguous()
        example_swiglu_bwd_kernel[(triton.cdiv(n,BLOCK),)](
            dact,gate,up,gate,up,n,BLOCK=BLOCK)
        del dact
        dx=torch.matmul(gate,wg)
        dx.add_(torch.matmul(up,wu))
        del wg,wu,gate,up
        _COUNTERS["bwd"]+=1
        return dx,None

def install_single_node_mlp():
    import bitsandbytes as bnb
    from transformers.models.qwen2 import modeling_qwen2
    cls=modeling_qwen2.Qwen2MLP
    previous=cls.forward
    if getattr(previous,"_soup_one_node_mlp",False):
        return True
    if not getattr(previous,"_soup_input_only_replay",False):
        raise RuntimeError("Input-only MLP parent required")
    def forward(self,x):
        if (not isinstance(x,torch.Tensor) or not x.is_cuda
                or x.dtype!=torch.float16 or not x.requires_grad
                or not torch.is_grad_enabled() or not self.training
                or x.ndim not in (2,3) or x.shape[-1]!=HIDDEN
                or not x.is_contiguous() or "qgroup" in DISABLED):
            _COUNTERS["fallback"]+=1
            return previous(self,x)
        group=save_up_group(self,x)
        down=getattr(self,"down_proj",None)
        base=getattr(down,"base_layer",down)
        tag=getattr(base,"_solution_group",None)
        if (group is None or group.checkpointed
                or group.split_now("fwd") or group.split_now("bwd")
                or type(base) is not bnb.nn.Linear4bit
                or down is not base
                or tag is None or tag[0] is not group or tag[1]!=DOWN
                or bool(base._forward_pre_hooks or base._forward_hooks or base._backward_hooks)
                or tuple(group.shapes[GATE])!=(INTERMEDIATE,HIDDEN)
                or tuple(group.shapes[UP])!=(INTERMEDIATE,HIDDEN)
                or tuple(group.shapes[DOWN])!=(HIDDEN,INTERMEDIATE)):
            _COUNTERS["fallback"]+=1
            return previous(self,x)
        y=_OneNodeMLP.apply(x,group)
        group.expected_bwd+=3
        _COUNTERS["engaged"]+=1
        return y
    forward._soup_one_node_mlp=True
    forward._previous=previous
    cls.forward=forward
    atexit.register(lambda:print("[own-one-node-mlp] "+repr(_COUNTERS)))
    return True
