import torch
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr, out_ptr,
    N, C, D1, D2, D3,
    OD1, OD2, OD3,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_D3: tl.constexpr,
):
    pid_rest = tl.program_id(0)
    pid_d3 = tl.program_id(1)

    od2 = pid_rest % OD2
    tmp = pid_rest // OD2
    od1 = tmp % OD1
    tmp = tmp // OD1
    c = tmp % C
    b = tmp // C

    offs_d3 = pid_d3 * BLOCK_D3 + tl.arange(0, BLOCK_D3)
    mask_d3 = offs_d3 < OD3

    base_d1 = od1 * STRIDE - PADDING
    base_d2 = od2 * STRIDE - PADDING
    base_d3 = offs_d3 * STRIDE - PADDING

    NEG_INF = -float('inf')
    result = tl.full((BLOCK_D3,), NEG_INF, tl.float32)

    nc_offset = (b * C + c) * D1 * D2 * D3

    interior = (od1 >= 1) & (od2 >= 1)

    if interior:
        for kd1 in range(KERNEL_SIZE):
            id1 = base_d1 + kd1 * DILATION
            for kd2 in range(KERNEL_SIZE):
                id2 = base_d2 + kd2 * DILATION
                in_base = nc_offset + (id1 * D2 + id2) * D3 + base_d3
                # kd3 = 0: still needs valid3 check
                valid3 = (base_d3 >= 0) & (base_d3 < D3)
                valid = valid3 & mask_d3
                val = tl.load(x_ptr + in_base, mask=valid, other=NEG_INF)
                result = tl.maximum(result, val)
                # kd3 = 1: only mask_d3 needed
                val = tl.load(x_ptr + in_base + DILATION, mask=mask_d3, other=NEG_INF)
                result = tl.maximum(result, val)
                # kd3 = 2: only mask_d3 needed
                val = tl.load(x_ptr + in_base + 2 * DILATION, mask=mask_d3, other=NEG_INF)
                result = tl.maximum(result, val)
    else:
        for kd1 in range(KERNEL_SIZE):
            id1 = base_d1 + kd1 * DILATION
            valid1 = (id1 >= 0) & (id1 < D1)
            for kd2 in range(KERNEL_SIZE):
                id2 = base_d2 + kd2 * DILATION
                valid2 = (id2 >= 0) & (id2 < D2)
                for kd3 in range(KERNEL_SIZE):
                    id3 = base_d3 + kd3 * DILATION
                    valid3 = (id3 >= 0) & (id3 < D3)
                    valid = valid1 & valid2 & valid3 & mask_d3
                    in_idx = nc_offset + (id1 * D2 + id2) * D3 + id3
                    val = tl.load(x_ptr + in_idx, mask=valid, other=NEG_INF)
                    result = tl.maximum(result, val)

    out_idx = (b * C + c) * (OD1 * OD2 * OD3) + (od1 * OD2 + od2) * OD3 + offs_d3
    tl.store(out_ptr + out_idx, result, mask=mask_d3)


def run(x):
    N, C, D1, D2, D3 = x.shape
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 3

    def out_dim(d):
        return (d + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    OD1 = out_dim(D1)
    OD2 = out_dim(D2)
    OD3 = out_dim(D3)

    out = torch.empty((N, C, OD1, OD2, OD3), device=x.device, dtype=x.dtype)

    BLOCK_D3 = 64
    grid = (N * C * OD1 * OD2, triton.cdiv(OD3, BLOCK_D3))
    maxpool3d_kernel[grid](
        x, out,
        N, C, D1, D2, D3,
        OD1, OD2, OD3,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        PADDING=padding,
        DILATION=dilation,
        BLOCK_D3=BLOCK_D3,
        num_warps=2,
    )
    return out
