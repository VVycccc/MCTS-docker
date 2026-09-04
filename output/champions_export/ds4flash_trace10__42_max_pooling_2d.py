import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=2),
    ],
    key=['H_out', 'W_out'],
)
@triton.jit
def max_pool2d_kernel(
    input_ptr, output_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    K: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = N * C * H_out * W_out
    mask = off < total_elements

    # Compute batch, channel, h_out, w_out from off
    batch = off // (C * H_out * W_out)
    rem = off - batch * (C * H_out * W_out)
    channel = rem // (H_out * W_out)
    rem2 = rem - channel * (H_out * W_out)
    h_out = rem2 // W_out
    w_out = rem2 - h_out * W_out

    # Compute input base offset for each element
    input_base = batch * (C * H * W) + channel * (H * W)

    # Initialize max_val to -inf
    max_val = tl.full([BLOCK_SIZE], -1e30, dtype=tl.float32)

    # Loop over kernel window
    for di in range(K):
        for dj in range(K):
            h_in = h_out * stride - padding + di * dilation
            w_in = w_out * stride - padding + dj * dilation
            # Check validity
            valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            # Compute offset
            offset = input_base + h_in * W + w_in
            # Load value, mask with valid and element mask
            val = tl.load(input_ptr + offset, mask=valid & mask, other=-1e30)
            # Update max
            max_val = tl.maximum(max_val, val)

    # Store output
    tl.store(output_ptr + off, max_val, mask=mask)


def run(x):
    N, C, H, W = x.shape
    kernel_size = 4
    stride = 1
    padding = 1
    dilation = 1
    H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    total_elements = N * C * H_out * W_out
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, output,
        N=N, C=C, H=H, W=W,
        K=kernel_size, stride=stride, padding=padding, dilation=dilation,
        H_out=H_out, W_out=W_out,
    )
    return output
