"""Independent Qwen2 within-batch packed decoder research.

Idea-level inspiration: Soup competitors demonstrate that packing valid tokens
can save decoder GEMM work if attention remains within each original example
and LoRA dropout consumes its native padded-size RNG stream. This file is an
original implementation, not a copy of any competitor's kernels or functions.
It uses the existing solution's original NF4/RMSNorm/streamed-loss kernels.
The second version implements independent row-segmented SDPA with explicit
failure on unsupported shape, rather than a quadratic dense block mask.
Physical T4 correctness, dropout RNG, and 64-step quality are mandatory gates.
"""
from contextvars import ContextVar
from functools import wraps

import torch

_LABEL_CONTEXT = ContextVar("soup_own_pack_labels", default=None)
_PACK_CONTEXT = ContextVar("soup_own_pack_geometry", default=None)
_SEGMENT_CONTEXT = ContextVar("soup_own_sdpa_segments", default=None)


def right_padded(mask):
    if (mask.ndim != 2 or mask.dtype not in
            (torch.int64, torch.int32, torch.bool, torch.uint8)):
        return False
    good = mask != 0
    if good.shape[0] < 2 or good.shape[1] < 2:
        return False
    return bool(good.any(-1).all().item() and
                not ((~good[:, :-1]) & good[:, 1:]).any().item())


def pack_segments(attention_mask):
    """Original padding compaction; preserve original within-batch token order."""
    batch, width = attention_mask.shape
    lengths = (attention_mask != 0).sum(dim=1).tolist()
    indices = torch.nonzero((attention_mask != 0).reshape(-1),
                            as_tuple=False).flatten()
    source_position = indices.remainder(width)
    bounds, offset = [], 0
    for length in lengths:
        length = int(length)
        bounds.append((offset, offset + length))
        offset += length
    assert offset == indices.numel()
    return indices, source_position, tuple(bounds)


class NativeShapedDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed_x, selected_bool_mask, scale):
        ctx.save_for_backward(selected_bool_mask)
        ctx.scale = scale
        return packed_x * (selected_bool_mask.to(packed_x.dtype) * scale)

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        return grad * (mask.to(grad.dtype) * ctx.scale), None, None


def patch_full_decoder_packing():
    """Patch narrow Qwen2 training-only paths after the normal Soup patches."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if getattr(Qwen2Model.forward, "_soup_competitor_research_packed", False):
        return True

    original_model = Qwen2Model.forward
    original_lm = Qwen2ForCausalLM.forward
    original_dropout = torch.nn.Dropout.forward
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    @wraps(original_sdpa)
    def segmented_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                            is_causal=False, scale=None, enable_gqa=False):
        plan = _SEGMENT_CONTEXT.get()
        if plan is None:
            return original_sdpa(query,key,value,attn_mask=attn_mask,
                                 dropout_p=dropout_p,is_causal=is_causal,
                                 scale=scale,enable_gqa=enable_gqa)
        bounds, expected, calls = plan
        # Never silently fall back to global-causal attention across examples.
        if (attn_mask is not None or not is_causal or dropout_p != 0
                or query.ndim != 4 or key.ndim != 4 or value.ndim != 4
                or query.shape[0] != 1 or key.shape[0] != 1
                or value.shape[0] != 1 or query.shape[-2] != expected
                or key.shape[-2] != expected or value.shape[-2] != expected):
            raise RuntimeError("Packed SDPA invariant failed: refusing cross-row attention")
        segments = []
        for left,right in bounds:
            segments.append(original_sdpa(
                query[:,:,left:right,:],key[:,:,left:right,:],
                value[:,:,left:right,:],attn_mask=None,dropout_p=0.0,
                is_causal=True,scale=scale,enable_gqa=enable_gqa))
        calls[0] += 1
        return torch.cat(segments,dim=-2)

    @wraps(original_dropout)
    def dropout_forward(module, x):
        plan = _PACK_CONTEXT.get()
        if (plan is None or id(module) not in plan[3]
                or type(module) is not torch.nn.Dropout
                or not module.training or module.p != 0.05 or module.inplace
                or not isinstance(x, torch.Tensor) or not x.is_cuda
                or x.dtype != torch.float32 or x.ndim != 3
                or tuple(x.shape[:2]) != (1, len(plan[2]))):
            return original_dropout(module, x)
        b, t, indices, _, counts = plan
        # Native CUDA Philox mask over the unchanged B,T,H logical geometry.
        # Only its selected entries are consumed by the packed adapter input.
        seed = torch.ones((b, t, x.shape[-1]), dtype=x.dtype, device=x.device)
        dummy, mask = torch.ops.aten.native_dropout(seed, module.p, True)
        del seed, dummy
        selected = mask.reshape(b * t, -1).index_select(0, indices).reshape_as(x)
        counts[0] += 1
        return NativeShapedDropout.apply(x, selected, 1.0 / (1.0 - module.p))

    @wraps(original_lm)
    def lm_forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        if (args or not self.training or not torch.is_grad_enabled()
                or not isinstance(labels, torch.Tensor) or not labels.is_cuda):
            return original_lm(self, *args, **kwargs)
        tok = _LABEL_CONTEXT.set(labels)
        try:
            return original_lm(self, *args, **kwargs)
        finally:
            _LABEL_CONTEXT.reset(tok)

    @wraps(original_model)
    def model_forward(self, input_ids=None, attention_mask=None,
                      position_ids=None, past_key_values=None,
                      inputs_embeds=None, use_cache=None, **kwargs):
        def native():
            return original_model(self, input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds,
                                  use_cache=use_cache, **kwargs)

        labels = _LABEL_CONTEXT.get()
        if (not self.training or not torch.is_grad_enabled()
                or not isinstance(input_ids, torch.Tensor)
                or not input_ids.is_cuda or input_ids.ndim != 2
                or not isinstance(attention_mask, torch.Tensor)
                or attention_mask.shape != input_ids.shape
                or attention_mask.device != input_ids.device
                or not isinstance(labels, torch.Tensor)
                or labels.shape != input_ids.shape
                or inputs_embeds is not None or past_key_values is not None
                or use_cache or self.gradient_checkpointing
                or getattr(self, "has_sliding_layers", False)
                or float(getattr(self.config, "attention_dropout", 0.0)) != 0.0
                or getattr(self.config, "_attn_implementation", "sdpa")
                   != "sdpa"
                or kwargs.get("output_hidden_states")
                or kwargs.get("output_attentions")
                or not right_padded(attention_mask)):
            return native()

        if not ((labels == -100) | (attention_mask != 0)).all().item():
            return native()
        b, t = input_ids.shape
        n_valid = int((attention_mask != 0).sum().item())
        if n_valid == b * t or n_valid == 0:
            return native()

        indices, pos, bounds = pack_segments(attention_mask)
        if position_ids is None:
            packed_position_ids = pos.view(1, -1)
        elif (isinstance(position_ids, torch.Tensor)
              and position_ids.ndim == 2 and position_ids.shape[-1] == t
              and position_ids.shape[0] in (1, b)):
            source_pos = position_ids.expand(b, -1).reshape(-1)
            packed_position_ids = source_pos.index_select(0, indices).view(1, -1)
        else:
            return native()

        allowed_dropout_ids = {
            id(layer) for name, layer in self.named_modules()
            if ".lora_dropout." in name and type(layer) is torch.nn.Dropout
            and layer.p == 0.05 and layer.training
        }
        if len(allowed_dropout_ids) != 2 * self.config.num_hidden_layers:
            return native()

        counts = [0]
        plan = (b, t, indices, allowed_dropout_ids, counts)
        sdpa_calls = [0]
        token = _PACK_CONTEXT.set(plan)
        sdpa_token = _SEGMENT_CONTEXT.set((bounds,len(indices),sdpa_calls))
        try:
            packed = original_model(
                self, input_ids=input_ids.reshape(-1).index_select(0, indices).view(1, -1),
                attention_mask={"full_attention": None},
                position_ids=packed_position_ids,
                past_key_values=None, inputs_embeds=None, use_cache=False,
                **kwargs,
            )
        finally:
            _PACK_CONTEXT.reset(token)
            _SEGMENT_CONTEXT.reset(sdpa_token)
        if sdpa_calls[0] != self.config.num_hidden_layers:
            raise RuntimeError("Packed SDPA activation count differs from decoder depth")
        if counts[0] != len(allowed_dropout_ids):
            raise RuntimeError("Native geometry dropout count differs from Q/V LoRA modules")

        # Restore the full original logical B,T layout for Soup loss/metrics.
        last = packed.last_hidden_state
        full = torch.zeros((b * t, last.shape[-1]), device=last.device, dtype=last.dtype)
        full.index_copy_(0, indices, last.reshape(-1, last.shape[-1]))
        self.__dict__["_soup_fullpack_last"] = {
            "real_tokens": len(indices), "original_tokens": b * t,
            "native_rng_dropout_calls": counts[0],
            "expected_dropout_calls": len(allowed_dropout_ids),
            "native_sdpa_calls": sdpa_calls[0],
            "attention_segments": tuple(tuple(pair) for pair in bounds),
        }
        return BaseModelOutputWithPast(
            last_hidden_state=full.view(b, t, -1),
            past_key_values=None, hidden_states=None, attentions=None,
        )

    Qwen2ForCausalLM.forward = lm_forward
    Qwen2Model.forward = model_forward
    torch.nn.Dropout.forward = dropout_forward
    torch.nn.functional.scaled_dot_product_attention = segmented_attention
    model_forward._soup_competitor_research_packed = True
    lm_forward._soup_competitor_research_packed = True
    dropout_forward._soup_competitor_research_packed = True
    segmented_attention._soup_competitor_research_packed = True
    return True
