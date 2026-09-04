import torch
import triton
import triton.language as tl


@triton.jit
def _maxpool2d_kernel(
    x_ptr, out_ptr,
    H, W, PH, PW,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)  # batch * channels 维

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_h = offs_h < PH
    mask_w = offs_w < PW

    acc = tl.full((BLOCK_H, BLOCK_W), float("-inf"), tl.float32)

    x_base = pid_nc * H * W

    for kh in range(0, KH):
        ih = offs_h * stride_h + kh * dil_h - pad_h
        ih_ok = (ih >= 0) & (ih < H)
        for kw in range(0, KW):
            iw = offs_w * stride_w + kw * dil_w - pad_w
            iw_ok = (iw >= 0) & (iw < W)
            ptrs = x_ptr + x_base + ih[:, None] * W + iw[None, :]
            m = ih_ok[:, None] & iw_ok[None, :]
            v = tl.load(ptrs, mask=m, other=float("-inf"))
            acc = tl.maximum(acc, v)

    out_ptrs = out_ptr + pid_nc * PH * PW + offs_h[:, None] * PW + offs_w[None, :]
    out_mask = mask_h[:, None] & mask_w[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def run(x):
    kernel_size, stride, padding, dilation = 4, 1, 1, 1
    x = x.contiguous()
    N, C, H, W = x.shape
    PH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    PW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out = torch.empty((N, C, PH, PW), device=x.device, dtype=x.dtype)

    BLOCK_H = 32
    BLOCK_W = 32
    grid = (triton.cdiv(PW, BLOCK_W), triton.cdiv(PH, BLOCK_H), N * C)
    _maxpool2d_kernel[grid](
        x, out,
        H, W, PH, PW,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        dilation, dilation,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out
