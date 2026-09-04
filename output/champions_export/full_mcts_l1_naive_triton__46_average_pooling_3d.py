import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=1),
    ],
    key=['W_OUT'],
)
@triton.jit
def avg_pool3d_kernel(
    x_ptr, y_ptr,
    NC, D, H, W,
    D_OUT, H_OUT, W_OUT,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_blocks = tl.cdiv(W_OUT, BLOCK_W)

    pid_w = pid % num_w_blocks
    pid_h = (pid // num_w_blocks) % H_OUT
    pid_nc_d = pid // (num_w_blocks * H_OUT)

    nc = pid_nc_d // D_OUT
    d_out = pid_nc_d % D_OUT
    h_out = pid_h
    w_out_base = pid_w * BLOCK_W

    w_offs = w_out_base + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W_OUT

    d_start = d_out * STRIDE - PADDING
    h_start = h_out * STRIDE - PADDING

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    HW = H * W
    DHW = D * HW

    for kd in range(0, KERNEL_SIZE):
        d = d_start + kd
        d_valid = (d >= 0) & (d < D)
        for kh in range(0, KERNEL_SIZE):
            h = h_start + kh
            h_valid = (h >= 0) & (h < H)
            valid_dh = d_valid & h_valid
            for kw in range(0, KERNEL_SIZE):
                w = w_offs * STRIDE - PADDING + kw
                w_valid = (w >= 0) & (w < W)
                valid = valid_dh & w_valid & w_mask
                x_offs = (nc.to(tl.int64) * DHW
                          + d.to(tl.int64) * HW
                          + h.to(tl.int64) * W
                          + w)
                x_val = tl.load(x_ptr + x_offs, mask=valid, other=0.0)
                acc += x_val

    result = acc / (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)

    y_hw = H_OUT * W_OUT
    y_dhw = D_OUT * y_hw
    y_offs = (nc.to(tl.int64) * y_dhw
              + d_out.to(tl.int64) * y_hw
              + h_out.to(tl.int64) * W_OUT
              + w_offs)
    tl.store(y_ptr + y_offs, result, mask=w_mask)


def run(x):
    batch_size, channels, depth, height, width = x.shape
    kernel_size = 3
    stride = 2
    padding = 1

    D_OUT = (depth + 2 * padding - kernel_size) // stride + 1
    H_OUT = (height + 2 * padding - kernel_size) // stride + 1
    W_OUT = (width + 2 * padding - kernel_size) // stride + 1

    y = torch.empty(batch_size, channels, D_OUT, H_OUT, W_OUT, device=x.device, dtype=x.dtype)

    NC = batch_size * channels
    grid = lambda meta: (NC * D_OUT * H_OUT * triton.cdiv(W_OUT, meta['BLOCK_W']),)

    avg_pool3d_kernel[grid](
        x, y,
        NC, depth, height, width,
        D_OUT, H_OUT, W_OUT,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        PADDING=padding,
    )
    return y
