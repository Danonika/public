"""Original guarded within-batch Qwen2 unpadding for the pinned T4 LoRA task.

Derived from Qwen2's public model geometry and PyTorch's ATen varlen-attention
contract, *not* the kernel source of any other challenge participant.

The model input, batch order, labels, and output shape are unchanged. The model
internals carry only active token rows; q/v LoRA independently draw dropout
masks on their ORIGINAL B*T dense shapes, then gather the active results.

Requires solution/varlen_gqa_attention.py, a separately authored component.
Experimental until real T4 + complete baseline-paired parity validates it.
"""

import contextvars
import atexit
import json
from functools import wraps
from importlib.metadata import version

import torch
import torch.nn.functional as F

from .varlen_gqa_attention import packed_gqa

_ACTIVE = contextvars.ContextVar("soup_ours_unpadded_forward", default=None)
_CURRENT_ATTENTION = contextvars.ContextVar("soup_ours_attention_projection", default=None)
_NATIVE_COUNT = {"packed_model_calls": 0, "varlen_calls": 0,
                 "q_dropout_dense": 0, "v_dropout_dense": 0}


class _RowPlan:
    def __init__(self, active, lengths, batch, width):
        self.active = active
        self.lengths = lengths
        self.batch = int(batch)
        self.width = int(width)
        self.tokens = int(active.numel())
        self.total = batch * width


def patch_original_qwen_unpad():
    """Install guarded research patch only for the pinned Qwen2/sm75 model."""
    if (not torch.cuda.is_available() or
        version("transformers") != "5.16.1" or
        version("peft") != "0.20.0"):
        return False

    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from peft.tuners.lora.bnb import Linear4bit as PeftLinear4bit

    if getattr(qwen.Qwen2Model.forward, "_soup_original_unpad", False):
        return True

    real_model_forward = qwen.Qwen2Model.forward
    real_attention_forward = qwen.Qwen2Attention.forward
    real_lora_forward = PeftLinear4bit.forward
    real_sdpa = F.scaled_dot_product_attention

    @wraps(real_sdpa)
    def row_isolated_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                          is_causal=False, scale=None, enable_gqa=False):
        plan = _ACTIVE.get()
        attn = _CURRENT_ATTENTION.get()
        if plan is None or attn is None:
            return real_sdpa(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal,
                             scale=scale, enable_gqa=enable_gqa)
        # Never allow a silent fallthrough to globally causal packed attention:
        # it would let a token see another batch item's keys.
        if (query.ndim != 4 or query.shape[0] != 1 or
            query.shape[-2] != plan.tokens or key.shape[-2] != plan.tokens or
            not query.is_cuda or query.device != key.device or
            query.device != value.device or dropout_p != 0.0 or
            (scale is not None and abs(float(scale) - query.shape[-1]**-0.5) > 1e-10)):
            raise RuntimeError("Unsupported packed attention: refuse cross-row leakage")
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
            raise RuntimeError("Packed attention requires original FP16 AMP mode")
        _NATIVE_COUNT["varlen_calls"] += 1
        return packed_gqa(query.to(torch.float16), key.to(torch.float16),
                          value.to(torch.float16), plan.lengths)

    @wraps(real_lora_forward)
    def full_geometry_dropout_lora(self, x, *args, **kwargs):
        plan = _ACTIVE.get()
        attn = _CURRENT_ATTENTION.get()
        if plan is None or attn is None or self is not attn.q_proj and self is not attn.v_proj:
            return real_lora_forward(self, x, *args, **kwargs)
        if (args or kwargs or not self.training or not isinstance(x,torch.Tensor)
            or not x.is_cuda or x.ndim != 3 or x.shape[:2] != (1,plan.tokens)):
            raise RuntimeError("Unsupported q/v packed LoRA; cannot change dropout RNG")
        # The original PEFT dropout samples B*T masks for q and v separately.
        # Running the exact original PEFT call on a freshly padded original-
        # geometry tensor preserves BOTH masks, RNG advancement and A/B grads.
        flat = torch.zeros((plan.total, x.shape[-1]), dtype=x.dtype,device=x.device)
        flat.index_copy_(0,plan.active,x.reshape(plan.tokens,-1))
        dense = flat.view(plan.batch,plan.width,x.shape[-1])
        result = real_lora_forward(self,dense)
        if result.shape[:2] != (plan.batch,plan.width):
            raise RuntimeError("Unexpected LoRA output geometry")
        name = "q_dropout_dense" if self is attn.q_proj else "v_dropout_dense"
        _NATIVE_COUNT[name] += 1
        return result.reshape(plan.total,-1).index_select(0,plan.active).view(1,plan.tokens,-1)

    @wraps(real_attention_forward)
    def routed_attention(self, *args, **kwargs):
        if _ACTIVE.get() is None:
            return real_attention_forward(self,*args,**kwargs)
        token = _CURRENT_ATTENTION.set(self)
        try:
            return real_attention_forward(self,*args,**kwargs)
        finally:
            _CURRENT_ATTENTION.reset(token)

    @wraps(real_model_forward)
    def packed_model(self, *args, **kwargs):
        # Only the single, standard, right-padded Qwen training shape is safe.
        if (args or not self.training or not torch.is_grad_enabled() or
            not torch.is_autocast_enabled("cuda") or
            torch.get_autocast_dtype("cuda") != torch.float16):
            return real_model_forward(self,*args,**kwargs)
        mask = kwargs.get("attention_mask")
        ids = kwargs.get("input_ids")
        embeds = kwargs.get("inputs_embeds")
        if (not isinstance(mask,torch.Tensor) or not mask.is_cuda or mask.ndim!=2
            or mask.shape[0] < 1 or mask.shape[1] < 2 or
            (ids is None and embeds is None) or (ids is not None and embeds is not None)
            or kwargs.get("past_key_values") is not None
            or kwargs.get("use_cache",False) or kwargs.get("output_attentions",False)
            or kwargs.get("output_hidden_states",False)
            or getattr(self,"gradient_checkpointing",False)
            or getattr(self.config,"_attn_implementation", "sdpa") != "sdpa"):
            return real_model_forward(self,*args,**kwargs)
        b,t = mask.shape
        if ids is not None and tuple(ids.shape)!=(b,t):
            return real_model_forward(self,*args,**kwargs)
        if embeds is not None and tuple(embeds.shape[:2])!=(b,t):
            return real_model_forward(self,*args,**kwargs)
        lengths = [int(x) for x in mask.to(torch.int32).sum(1).tolist()]
        if min(lengths) < 1 or max(lengths) > t or sum(lengths)==b*t:
            return real_model_forward(self,*args,**kwargs)
        expected=(torch.arange(t,device=mask.device)[None,:]
                  < torch.tensor(lengths,device=mask.device)[:,None])
        if not torch.equal(mask!=0,expected):
            return real_model_forward(self,*args,**kwargs)
        # Preserve the original dense position_id values at the active rows.
        base_pos=kwargs.get("position_ids")
        if base_pos is not None and (not isinstance(base_pos,torch.Tensor) or
            base_pos.ndim!=2 or base_pos.shape[1]!=t or
            base_pos.shape[0] not in (1,b)):
            return real_model_forward(self,*args,**kwargs)
        dense_embeds = embeds if embeds is not None else self.embed_tokens(ids)
        active = torch.nonzero(mask.reshape(-1)!=0,as_tuple=False).flatten()
        compact = dense_embeds.reshape(b*t,-1).index_select(0,active).unsqueeze(0)
        if base_pos is None:
            pos=torch.cat([torch.arange(n,device=mask.device,dtype=torch.long) for n in lengths]).view(1,-1)
        elif base_pos.shape[0]==1:
            pos=base_pos.expand(b,-1).reshape(-1).index_select(0,active).view(1,-1)
        else:
            pos=base_pos.reshape(-1).index_select(0,active).view(1,-1)
        plan = _RowPlan(active,lengths,b,t)
        start = _NATIVE_COUNT["varlen_calls"]
        token = _ACTIVE.set(plan)
        try:
            out = real_model_forward(
                self, input_ids=None, attention_mask=None, inputs_embeds=compact,
                position_ids=pos, past_key_values=None, use_cache=False,
                **{k:v for k,v in kwargs.items() if k not in
                   ("input_ids","attention_mask","inputs_embeds","position_ids",
                    "past_key_values","use_cache")},
            )
            # One independent row-isolated attention per Qwen decoder layer.
            if _NATIVE_COUNT["varlen_calls"]-start != len(self.layers):
                raise RuntimeError("Packed Qwen2 attention did not invoke row-isolated SDPA for every layer")
        finally:
            _ACTIVE.reset(token)
        flat=torch.zeros((b*t, out.last_hidden_state.shape[-1]),
                         dtype=out.last_hidden_state.dtype,device=out.last_hidden_state.device)
        flat.index_copy_(0,active,out.last_hidden_state.reshape(plan.tokens,-1))
        _NATIVE_COUNT["packed_model_calls"] += 1
        return BaseModelOutputWithPast(last_hidden_state=flat.view(b,t,-1),
                                       past_key_values=None)

    full_geometry_dropout_lora._soup_original_unpad=True
    row_isolated_sdpa._soup_original_unpad=True
    packed_model._soup_original_unpad=True
    PeftLinear4bit.forward=full_geometry_dropout_lora
    qwen.Qwen2Attention.forward=routed_attention
    F.scaled_dot_product_attention=row_isolated_sdpa
    qwen.Qwen2Model.forward=packed_model
    return True


def _report():
    if _NATIVE_COUNT["packed_model_calls"]:
        print("[soup_original_packed_rows] " + json.dumps(_NATIVE_COUNT, sort_keys=True), flush=True)


atexit.register(_report)