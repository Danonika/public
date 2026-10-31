"""Qwen pad-elision experiment for the PEFT q_proj only.

Valid-row projection uses original bitsandbytes matmul and LoRA math; dropout
is sampled on full B*T geometry to conserve the generator state and masks.
"""
import torch

def patch_dual_qv_projection_compaction():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    from peft.tuners.lora.bnb import Linear4bit
    from .mlp_padding import _ACTIVE_ROWS
    if getattr(Qwen2Attention.forward, "_soup_qv_padding", False):
        return True

    original_attention_forward = Qwen2Attention.forward
    original_lora_forward = Linear4bit.forward

    def compact_query_forward(self, x, *args, **kwargs):
        plan = getattr(self, "_soup_qv_padding_plan", None)
        if plan is None:
            return original_lora_forward(self, x, *args, **kwargs)
        active, mask_shape = plan[0], plan[1]
        if (args or kwargs or not self.training or self.disable_adapters or self.merged
                or not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_cuda
                or tuple(x.shape[:2]) != mask_shape or active.device != x.device
                or active.ndim != 1 or active.dtype != torch.long
                or active.numel() == 0 or active.numel() == x.shape[0]*x.shape[1]
                or len(self.active_adapters) != 1):
            return original_lora_forward(self, x, *args, **kwargs)
        adapter = self.active_adapters[0]
        if adapter not in self.lora_A or adapter in self.lora_variant:
            return original_lora_forward(self, x, *args, **kwargs)
        self._check_forward_args(x, *args, **kwargs)
        dim=x.shape[-1]
        compact_x=x.reshape(-1,dim).index_select(0,active)
        base=self.base_layer(compact_x).clone()
        lora_A=self.lora_A[adapter]
        lora_B=self.lora_B[adapter]
        dropout=self.lora_dropout[adapter]
        scale=self.scaling[adapter]
        requires_conversion=not torch.is_autocast_enabled()
        dropout_x=self._cast_input_dtype(x,lora_A.weight.dtype) if requires_conversion else x
        # Do not generate dropout on compact x: the full-mask RNG sequence is required.
        dropout_full=dropout(dropout_x)
        dropout_live=dropout_full.reshape(-1,dim).index_select(0,active)
        lora_delta=lora_B(lora_A(dropout_live))*scale
        if requires_conversion:
            lora_delta=lora_delta.to(base.dtype)
        live=base+lora_delta
        flat=torch.zeros((x.shape[0]*x.shape[1],live.shape[-1]),dtype=live.dtype,device=live.device)
        flat=flat.index_copy(0,active,live)
        return flat.view(x.shape[0],x.shape[1],live.shape[-1])

    def attention_forward(self, *args, **kwargs):
        plan = _ACTIVE_ROWS.get()
        q = self.q_proj
        v = self.v_proj
        if (not self.training or plan is None
                or not isinstance(q, Linear4bit)
                or not isinstance(v, Linear4bit)
                or not isinstance(plan, tuple) or len(plan) < 2
                or getattr(q, "_soup_qv_padding_plan", None) is not None
                or getattr(v, "_soup_qv_padding_plan", None) is not None):
            return original_attention_forward(self, *args, **kwargs)
        q._soup_qv_padding_plan = plan
        v._soup_qv_padding_plan = plan
        try:
            return original_attention_forward(self, *args, **kwargs)
        finally:
            del q._soup_qv_padding_plan
            del v._soup_qv_padding_plan

    compact_query_forward._soup_qv_padding=True
    attention_forward._soup_qv_padding=True
    Linear4bit.forward=compact_query_forward
    Qwen2Attention.forward=attention_forward
    return True
