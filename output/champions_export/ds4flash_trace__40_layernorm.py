import torch
import torch.nn as nn  # kept for interface compatibility, not used
import triton
import triton.language as tl

_weights_path = "problems/kb_level1/40_layernorm_weights.pt"
_W = None
_B = None


@triton.jit
def partial_sum_kernel(
    X, Sum, SumSq, D, BLOCK_SIZE: tl.constexpr
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_d = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_d < D
    x = tl.load(X + pid_b * D + offs_d, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    sq = tl.sum(x * x, axis=0)
    tl.atomic_add(Sum + pid_b, s)
    tl.atomic_add(SumSq + pid_b, sq)


@triton.jit
def normalize_kernel(
    X, Y, W, Bw, Sum, SumSq, D, B, eps, BLOCK_SIZE: tl.constexpr
):
    pid_d = tl.program_id(0)
    offs_d = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_d < D
    w = tl.load(W + offs_d, mask=mask, other=0.0)
    b = tl.load(Bw + offs_d, mask=mask, other=0.0)
    for i in range(0, B):
        mean = tl.load(Sum + i) / D
        var = tl.load(SumSq + i) / D - mean * mean
        inv_std = 1.0 / tl.sqrt(var + eps)
        x = tl.load(X + i * D + offs_d, mask=mask, other=0.0)
        y = (x - mean) * inv_std * w + b
        tl.store(Y + i * D + offs_d, y, mask=mask)


def run(x):
    global _W, _B

    if _W is None:
        state = torch.load(_weights_path, map_location="cpu", weights_only=True)
        _W = state["ln.weight"].flatten().contiguous()
        _B = state["ln.bias"].flatten().contiguous()

    if _W.device != x.device:
        _W = _W.to(x.device)
        _B = _B.to(x.device)

    x = x.contiguous()
    B = x.shape[0]
    D = x.numel() // B
    y = torch.empty_like(x)

    sums = torch.zeros(B, device=x.device, dtype=torch.float32)
    sumsqs = torch.zeros(B, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 2048
    grid1 = (triton.cdiv(D, BLOCK_SIZE), B)
    partial_sum_kernel[grid1](x, sums, sumsqs, D, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    grid2 = (triton.cdiv(D, BLOCK_SIZE),)
    normalize_kernel[grid2](x, y, _W, _B, sums, sumsqs, D, B, 1e-5, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return y
