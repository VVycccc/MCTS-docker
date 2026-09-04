import torch
import triton
import triton.language as tl

_W = None
_W_DEVICE = None

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N,
    stride_x_batch,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    x_row_ptr = x_ptr + pid * stride_x_batch
    y_row_ptr = y_ptr + pid * stride_x_batch

    # First pass: accumulate sum and sum of squares (serial reduction)
    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_row_ptr + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)

    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, apply weight & bias, store
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_row_ptr + idx, mask=mask, other=0.0)
        w = tl.load(w_ptr + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0)
        y = (x - mean) * inv_std * w + b
        tl.store(y_row_ptr + idx, y, mask=mask)


def run(x):
    global _W, _W_DEVICE
    if _W is None or _W_DEVICE != str(x.device):
        sd = torch.load(
            "/home/wangyichen/DirecTune-MCTS/problems/kb_level1/40_layernorm_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        _W = (sd["ln.weight"].to(x.device), sd["ln.bias"].to(x.device))
        _W_DEVICE = str(x.device)
    weight, bias = _W

    N = 1
    for d in x.shape[1:]:
        N *= d
    batch = x.shape[0]
    eps = 1e-5

    y = torch.empty_like(x)
    grid = (batch,)
    layernorm_kernel[grid](x, weight, bias, y, N, x.stride(0), eps)
    return y
