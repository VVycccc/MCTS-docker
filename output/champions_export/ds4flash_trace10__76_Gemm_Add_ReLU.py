import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/76_Gemm_Add_ReLU_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = torch.load(_weights_path, map_location='cpu', weights_only=True)
    _weights = {k: v.to(device) for k, v in _weights.items()}
    _weights['gemm.weight'] = _weights['gemm.weight'].half()
    _device = str(device)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_add_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Load a: (BLOCK_M, BLOCK_K)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=a_mask, other=0.0)
        # Load b transposed: (BLOCK_K, BLOCK_N) directly from column-major view
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=True)

    # Bias and ReLU
    bias_offs = offs_n
    bias_mask = bias_offs < N
    bias = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0)
    acc += bias[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=c_mask)

def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)
    w = _weights['gemm.weight']   # (N, K) half
    bias = _weights['bias']       # (N,) float32
    M, K = x.shape
    N = w.shape[0]
    x_half = x.half()
    c = torch.empty((M, N), dtype=torch.float32, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_add_relu_kernel[grid](
        x_half, w, bias, c,
        M, N, K,
        x_half.stride(0), x_half.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
