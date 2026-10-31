"""Independent original Triton NF4 wordpack decoder; benchmark-only experimental overlay.

Computes each packed NF4 byte in a flat 1-D tile and writes two FP16 values
packed into a 32-bit word. Native bitsandbytes double-quant order is preserved.
No competitor kernel implementation was copied. Guards fall back to the
preexisting known-good launch for unsupported quant geometry or device.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def soup_original_nf4_wordflat(
    packed, codes8, nested_lut, nested_abs, nested_offset, values16, out_word,
    BLOCKS: tl.constexpr, QSIZE: tl.constexpr, Q2SIZE: tl.constexpr
):
    # Flat per-byte coordinates, not a 2D block×byte tile.
    lane = tl.program_id(0) * (BLOCKS * (QSIZE // 2)) + tl.arange(
        0, BLOCKS * (QSIZE // 2))
    scale_row = lane // (QSIZE // 2)
    secondary_row = scale_row // Q2SIZE
    ix = tl.load(codes8 + scale_row).to(tl.int32)
    x = tl.load(nested_lut + ix).to(tl.float32)
    y = tl.load(nested_abs + secondary_row).to(tl.float32)
    z = tl.load(nested_offset + secondary_row).to(tl.float32)
    scaled = x * y
    scaled = scaled + z
    raw = tl.load(packed + lane).to(tl.int32)
    lo_code = raw >> 4
    hi_code = raw & 15
    lo = (tl.load(values16 + lo_code).to(tl.float32) * scaled).to(tl.float16)
    hi = (tl.load(values16 + hi_code).to(tl.float32) * scaled).to(tl.float16)
    lo_bits = lo.to(tl.uint16, bitcast=True).to(tl.uint32)
    hi_bits = hi.to(tl.uint16, bitcast=True).to(tl.uint32)
    tl.store(out_word + lane, lo_bits | (hi_bits << 16))


def install_wordflat_group_decode():
    from . import qgroup
    if getattr(qgroup.launch, "_soup_original_flatword", False):
        return True
    original = qgroup.launch

    def launch(packed, absmax8, code, absmax2, offset2,
               levels, out, n_rows, qblock, qblock2):
        supported = (
            qblock == 64 and qblock2 == 256 and
            n_rows > 0 and n_rows % 64 == 0 and
            packed.is_cuda and packed.is_contiguous() and
            absmax8.is_cuda and absmax2.is_cuda and
            levels.is_cuda and offset2.is_cuda and
            out.dtype == torch.float16 and out.is_contiguous()
        )
        if not supported:
            return original(packed,absmax8,code,absmax2,offset2,
                            levels,out,n_rows,qblock,qblock2)
        soup_original_nf4_wordflat[(n_rows // 64,)](
            packed, absmax8, code, absmax2, offset2,
            levels, out.view(torch.int32), BLOCKS=64, QSIZE=64,
            Q2SIZE=256, num_warps=4, enable_fp_fusion=False
        )

    launch._soup_original_flatword=True
    qgroup.launch = launch
    return True
