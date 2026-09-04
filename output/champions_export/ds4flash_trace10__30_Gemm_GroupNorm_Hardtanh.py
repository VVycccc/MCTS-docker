import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Weight caching (module-level, loaded once per device)
# ---------------------------------------------------------------------------
_WEIGHTS_PATH = "/home/wangyichen/DirecTune/problems/kb_level2/30_Gemm_GroupNorm_Hardtanh_weights.pt"
_WEIGHTS = None
_WEIGHT_DEVICE = None

def _load_weights(device):
    global _WEIGHTS, _WEIGHT_DEVICE
    if _WEIGHTS is None or _WEIGHT_DEVICE != str(device):
        raw = torch.load(_WEIGHTS_PATH, map_location='cpu', weights_only=True)
        _WEIGHTS = {k: v.to(device) for k, v in raw.items()}
        _WEIGHT_DEVICE = str(device)

# ---------------------------------------------------------------------------
# GEMM kernel (no Tensor Core, no autotune, small tile)
# ---------------------------------------------------------------------------
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # store
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=mask_c)


# ---------------------------------------------------------------------------
# GroupNorm + HardTanh kernel (two-pass, block-size=group-size, no mask)
# ---------------------------------------------------------------------------
@triton.jit
def group_norm_hardtanh_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    batch_size, C, G, eps,
    hardtanh_min: tl.constexpr, hardtanh_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)   # batch index
    pid_g = tl.program_id(1)   # group index

    # base offset for this group
    base = pid_b * C + pid_g * BLOCK
    offs = tl.arange(0, BLOCK)

    # ---- First pass: compute sum and sum_sq ----
    x = tl.load(x_ptr + base + offs)
    sum_x = tl.sum(x, axis=0)
    sum_xx = tl.sum(x * x, axis=0)
    mean = sum_x / BLOCK
    var = sum_xx / BLOCK - mean * mean
    inv_std = tl.rsqrt(var + eps)

    # ---- Second pass: normalize, apply weight/bias, hardtanh ----
    x = tl.load(x_ptr + base + offs)
    w = tl.load(w_ptr + pid_g * BLOCK + offs)
    bias = tl.load(b_ptr + pid_g * BLOCK + offs)
    normalized = (x - mean) * inv_std
    normalized = normalized * w + bias
    # HardTanh
    normalized = tl.where(normalized < hardtanh_min, hardtanh_min, normalized)
    normalized = tl.where(normalized > hardtanh_max, hardtanh_max, normalized)
    tl.store(y_ptr + base + offs, normalized)


# ---------------------------------------------------------------------------
# Public entry point (same signature as reference run)
# ---------------------------------------------------------------------------
def run(x):
    # ----- weight loading -----
    _load_weights(x.device)
    gemm_weight = _WEIGHTS['gemm.weight']          # [8192, 8192]
    gemm_bias = _WEIGHTS['gemm.bias']               # [8192]
    gn_weight = _WEIGHTS['group_norm.weight']       # [8192]
    gn_bias = _WEIGHTS['group_norm.bias']           # [8192]

    # ----- dimensions -----
    batch_size, in_features = x.shape
    out_features = gemm_weight.shape[0]    # 8192
    N = out_features
    K = in_features
    M = batch_size

    # ------------------------------------------------------------------
    # Step 1: GEMM  (x @ W^T + bias)
    # ------------------------------------------------------------------
    # Convert inputs to fp16 for Tensor Core acceleration
    x_fp16 = x.to(torch.float16)
    w_fp16 = gemm_weight.to(torch.float16)
    gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid_gemm = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_kernel[grid_gemm](
        x_fp16, w_fp16, gemm_out, gemm_bias,
        M, N, K,
        x_fp16.stride(0), x_fp16.stride(1),
        w_fp16.stride(1), w_fp16.stride(0),
        gemm_out.stride(0), gemm_out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    # ------------------------------------------------------------------
    # Step 2: GroupNorm + HardTanh
    # ------------------------------------------------------------------
    num_groups = 16
    group_size = N // num_groups          # 512
    eps = 1e-5
    hardtanh_min = -2.0
    hardtanh_max = 2.0

    gn_out = torch.empty_like(gemm_out)
    # each block processes one batch × one group
    grid_gn = (batch_size, num_groups)
    BLOCK_GN = 512   # group size, also BLOCK for kernel
    group_norm_hardtanh_kernel[grid_gn](
        gemm_out, gn_out, gn_weight, gn_bias,
        batch_size, N, num_groups, eps,
        hardtanh_min=hardtanh_min, hardtanh_max=hardtanh_max,
        BLOCK=BLOCK_GN,
    )

    return gn_out
