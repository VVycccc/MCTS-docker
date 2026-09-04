import torch
import triton
import triton.language as tl

@triton.jit
def avg_pool_1d_kernel(
    x_ptr, y_ptr,
    in_channels, input_length, output_length,
    stride, padding,
    KERNEL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_bc = tl.program_id(1)

    batch = pid_bc // in_channels
    channel = pid_bc % in_channels

    offs = pid_o * BLOCK + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, KERNEL_SIZE)

    base = batch * in_channels * input_length + channel * input_length
    base_y = batch * in_channels * output_length + channel * output_length

    in_pos = offs[:, None] * stride + offs_k[None, :] - padding

    tile_start = pid_o * BLOCK
    in_lo = tile_start * stride - padding
    in_hi = (tile_start + BLOCK - 1) * stride + KERNEL_SIZE - 1 - padding
    out_hi = tile_start + BLOCK

    if (in_lo >= 0) & (in_hi < input_length) & (out_hi <= output_length):
        x_val = tl.load(x_ptr + base + in_pos)
        acc = tl.sum(x_val, axis=1)
        tl.store(y_ptr + base_y + offs, acc / KERNEL_SIZE)
    else:
        mask = offs < output_length
        in_mask = (in_pos >= 0) & (in_pos < input_length) & mask[:, None]
        x_val = tl.load(x_ptr + base + in_pos, mask=in_mask, other=0.0)
        acc = tl.sum(x_val, axis=1)
        tl.store(y_ptr + base_y + offs, acc / KERNEL_SIZE, mask=mask)


def run(x):
    batch_size, in_channels, input_length = x.shape
    kernel_size = 8
    stride = 1
    padding = 4
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1

    y = torch.empty((batch_size, in_channels, output_length), device=x.device, dtype=x.dtype)

    BLOCK = 1024
    grid = (triton.cdiv(output_length, BLOCK), batch_size * in_channels)
    avg_pool_1d_kernel[grid](
        x, y,
        in_channels, input_length, output_length,
        stride, padding,
        KERNEL_SIZE=kernel_size,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return y
