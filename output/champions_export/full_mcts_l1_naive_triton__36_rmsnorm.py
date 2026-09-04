import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr, out_ptr,
    n_features,
    batch_size, dim1, dim2,
    stride_b, stride_f, stride_i, stride_j,
    eps,
    BLOCK_F: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)

    num_j_groups = tl.cdiv(dim2, BLOCK_J)
    num_ij = dim1 * num_j_groups
    b = pid // num_ij
    rem = pid % num_ij
    i = rem // num_j_groups
    j_start = (rem % num_j_groups) * BLOCK_J

    offs_f = tl.arange(0, BLOCK_F)
    offs_j = j_start + tl.arange(0, BLOCK_J)
    mask_f = offs_f < n_features
    mask_j = offs_j < dim2

    base = b * stride_b + i * stride_i
    x = tl.load(
        x_ptr + base + offs_f[:, None] * stride_f + offs_j[None, :] * stride_j,
        mask=mask_f[:, None] & mask_j[None, :],
        other=0.0,
    )

    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / n_features + eps)

    out = x * inv_rms[None, :]
    tl.store(
        out_ptr + base + offs_f[:, None] * stride_f + offs_j[None, :] * stride_j,
        out,
        mask=mask_f[:, None] & mask_j[None, :],
    )


def run(x):
    batch_size, features, dim1, dim2 = x.shape
    out = torch.empty_like(x)

    BLOCK_F = 64
    BLOCK_J = 128
    num_j_groups = triton.cdiv(dim2, BLOCK_J)
    total = batch_size * dim1 * num_j_groups
    grid = (total,)

    rmsnorm_kernel[grid](
        x, out,
        features,
        batch_size, dim1, dim2,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        1e-5,
        BLOCK_F=BLOCK_F,
        BLOCK_J=BLOCK_J,
    )
    return out
