import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]


@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    B, C, H, W,
    OH, OW,
    stride, padding, dilation,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C

    oh_offs = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)

    # 预计算 base offset，减少循环内重复计算
    x_bc_offset = b * C * H * W + c * H * W
    out_bc_offset = b * C * OH * OW + c * OH * OW

    max_val = tl.full((BLOCK_OH, BLOCK_OW), float('-inf'), dtype=tl.float32)

    for kh in range(KERNEL_SIZE):
        for kw in range(KERNEL_SIZE):
            ih = oh_offs * stride - padding + kh * dilation
            iw = ow_offs * stride - padding + kw * dilation

            ih_2d = ih[:, None]
            iw_2d = iw[None, :]

            valid = (ih_2d >= 0) & (ih_2d < H) & (iw_2d >= 0) & (iw_2d < W)

            x_offs = x_bc_offset + ih_2d * W + iw_2d

            x_vals = tl.load(x_ptr + x_offs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, x_vals)

    out_offs = out_bc_offset + oh_offs[:, None] * OW + ow_offs[None, :]
    out_mask = (oh_offs[:, None] < OH) & (ow_offs[None, :] < OW)
    tl.store(out_ptr + out_offs, max_val, mask=out_mask)


def run(x):
    B, C, H, W = x.shape
    ks, st, pd, dl = get_init_inputs()
    OH = (H + 2 * pd - dl * (ks - 1) - 1) // st + 1
    OW = (W + 2 * pd - dl * (ks - 1) - 1) // st + 1
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OH = 64
    BLOCK_OW = 64
    grid = (B * C, triton.cdiv(OH, BLOCK_OH), triton.cdiv(OW, BLOCK_OW))
    maxpool2d_kernel[grid](
        x, out,
        B, C, H, W,
        OH, OW,
        st, pd, dl,
        KERNEL_SIZE=ks,
        BLOCK_OH=BLOCK_OH,
        BLOCK_OW=BLOCK_OW,
        num_warps=8,
    )
    return out
