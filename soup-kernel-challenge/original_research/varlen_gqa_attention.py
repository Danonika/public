"""Original T4 packed causal GQA attention proof-of-concept using native PyTorch.

This module tests the *idea* of flat within-batch sequences and grouped-query
attention with the built-in ATen efficient attention op; no competitor kernel or
training code is copied. It is not yet integrated into Soup training.
"""

import torch
from torch.autograd import Function


class _PackedCausalAttention(Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_len, scale):
        if q.ndim != 4 or q.shape[0] != 1 or q.dtype != torch.float16:
            raise ValueError("Expected [1, N, H, D] fp16 packed attention input")
        if k.shape != v.shape or k.shape != q.shape or not q.is_cuda:
            raise ValueError("Expanded GQA must share packed shape and device")
        if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_cuda:
            raise ValueError("CUDA int32 row boundaries required")
        if max_len <= 0 or scale <= 0:
            raise ValueError("Invalid attention geometry")
        out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
            q.contiguous(), k.contiguous(), v.contiguous(), None,
            cu_seqlens, cu_seqlens, max_len, max_len,
            0.0, 1, True, scale=scale,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens, seed, offset)
        ctx.max_len = max_len
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, out, lse, boundaries, seed, offset = ctx.saved_tensors
        dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
            grad_output.contiguous(), q, k, v, None, out, boundaries, boundaries,
            ctx.max_len, ctx.max_len, lse, 0.0, seed, offset, 1, False,
            scale=ctx.scale,
        )
        return dq, dk, dv, None, None, None


def packed_gqa(q, k, v, lengths):
    """Q,K,V: [1, heads, sum(lengths), head_dim] in the original head layout.

    The cache heads are explicitly expanded *within the existing batch*, not
    across independent sequences; autograd's repeat_interleave backward sums
    their gradients into the two real KV heads. No library other than PyTorch.
    """
    if len(lengths) == 0 or min(lengths) <= 0:
        raise ValueError("Each packed example must contain real tokens")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Expected 4D packed B=1 QKV")
    if q.shape[2] != sum(lengths) or k.shape[2] != sum(lengths):
        raise ValueError("Packed count and cumulative boundary counts disagree")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K/V geometry mismatch")
    ratio = q.shape[1] // k.shape[1]
    if ratio <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid GQA head ratio")
    cu = torch.tensor([0] + list(_prefix(lengths)), device=q.device, dtype=torch.int32)
    qwork = q.transpose(1, 2).contiguous()
    kwork = k.transpose(1, 2).repeat_interleave(ratio, dim=2).contiguous()
    vwork = v.transpose(1, 2).repeat_interleave(ratio, dim=2).contiguous()
    result = _PackedCausalAttention.apply(qwork, kwork, vwork, cu, max(lengths), q.shape[-1] ** -0.5)
    return result.transpose(1, 2)


def _prefix(lengths):
    total = 0
    for n in lengths:
        total += int(n)
        yield total


def test_synthetic_t4():
    import torch.nn.functional as F
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    torch.manual_seed(31907)
    results = []
    for lengths in ([1,2,3,4], [6,11,19,7], [49,112,63,31], [8,16,32,64]):
        for shape in ((12,2,128),(12,2,64)):
            Hq,Hkv,D=shape
            N=sum(lengths)
            q=torch.randn((1,Hq,N,D),device="cuda",dtype=torch.float16,requires_grad=True)
            k=torch.randn((1,Hkv,N,D),device="cuda",dtype=torch.float16,requires_grad=True)
            v=torch.randn((1,Hkv,N,D),device="cuda",dtype=torch.float16,requires_grad=True)
            qo=q.detach().clone().requires_grad_(True)
            ko=k.detach().clone().requires_grad_(True)
            vo=v.detach().clone().requires_grad_(True)
            ours=packed_gqa(q,k,v,lengths)
            refs=[]; start=0
            for count in lengths:
                end=start+count
                qi=qo[:,:,start:end,:]
                ki=ko[:,:,start:end,:].repeat_interleave(Hq//Hkv,dim=1)
                vi=vo[:,:,start:end,:].repeat_interleave(Hq//Hkv,dim=1)
                refs.append(F.scaled_dot_product_attention(qi,ki,vi,dropout_p=0.0,is_causal=True))
                start=end
            reference=torch.cat(refs,dim=2)
            torch.testing.assert_close(ours,reference,atol=0.009,rtol=0.009)
            dy=torch.randn_like(ours)
            grads=torch.autograd.grad(ours,(q,k,v),dy)
            refgrads=torch.autograd.grad(reference,(qo,ko,vo),dy)
            for item1,item2 in zip(grads,refgrads):
                torch.testing.assert_close(item1,item2,atol=0.02,rtol=0.02)
            results.append({"N":N,"D":D,"Hq":Hq,"Hkv":Hkv,
                            "fwd_abs":float((ours-reference).abs().max().item()),
                            "grad_abs":max(float((a-b).abs().max().item()) for a,b in zip(grads,refgrads))})
    return results