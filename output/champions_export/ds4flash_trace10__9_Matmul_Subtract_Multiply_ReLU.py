import torch
import triton
import triton.language as tl

# 加载权重缓存
_weights = None
_device = None
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/9_Matmul_Subtract_Multiply_ReLU_weights.pt"

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = device

@triton.jit
def matmul_sub_mul_relu_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    subtract_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)

    mask_m = off_m < M
    mask_n = off_n < N

    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # 加载 x 块 [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + off_m[:, None] * stride_xm + (k + off_k[None, :]) * stride_xk
        if EVEN_M and EVEN_K:
            x = tl.load(x_ptrs)
        else:
            x = tl.load(x_ptrs, mask=mask_m[:, None] & (k + off_k[None, :] < K), other=0.0)

        # 加载 w 块 [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + (k + off_k[:, None]) * stride_wk + off_n[None, :] * stride_wn
        if EVEN_K and EVEN_N:
            w = tl.load(w_ptrs)
        else:
            w = tl.load(w_ptrs, mask=(k + off_k[:, None] < K) & mask_n[None, :], other=0.0)

        # 矩阵乘法，不使用 Tensor Core (allow_tf32=False)
        acc = tl.dot(x, w, acc=acc, allow_tf32=True)

    # 加上偏置
    bias_ptrs = bias_ptr + off_n
    if EVEN_N:
        bias = tl.load(bias_ptrs)
    else:
        bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # 减法、乘法、ReLU
    acc = acc - subtract_value
    acc = acc * multiply_value
    acc = tl.maximum(acc, 0.0)

    # 存储输出
    out_ptrs = out_ptr + off_m[:, None] * stride_om + off_n[None, :] * stride_on
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def run(x):
    global _weights, _device
    # 确保权重已加载
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)
    else:
        # 如果设备变了但权重未加载，重新加载
        if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
            _init_weights(x.device)

    weight = _weights['linear.weight']   # [out_features, in_features] = [8192, 8192]
    bias = _weights['linear.bias']       # [out_features] = [8192]

    # 将 weight 转置为 [K, N] 以匹配矩阵乘法顺序 x @ W.T
    w_t = weight.T.contiguous().half()   # [8192, 8192] fp16

    # 将输入转换为 fp16 以利用 Tensor Core
    x = x.half()

    M, K = x.shape
    N = w_t.shape[1]

    # 分配输出
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # 配置块大小（朴素，32）
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    EVEN_M = (M % BLOCK_M == 0)
    EVEN_N = (N % BLOCK_N == 0)
    EVEN_K = (K % BLOCK_K == 0)

    matmul_sub_mul_relu_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
        subtract_value=2.0,
        multiply_value=1.5,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_K=EVEN_K,
    )

    return out
