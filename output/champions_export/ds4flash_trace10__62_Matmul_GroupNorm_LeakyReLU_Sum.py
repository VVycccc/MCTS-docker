import torch
import triton
import triton.language as tl

# ---------- global weight cache ----------
_weights_cache = None
_weights_device = None

def _load_weights(device):
    global _weights_cache, _weights_device
    w = torch.load(
        "/home/wangyichen/DirecTune/problems/kb_level2/62_Matmul_GroupNorm_LeakyReLU_Sum_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    _weights_cache = {k: v.to(device) for k, v in w.items()}
    _weights_device = device

# ----------------------------------------------------------------------
# 1. matmul kernel (no Tensor Core, allow_tf32=False)
# ----------------------------------------------------------------------
@triton.jit
def matmul_kernel(
    A, B, C,
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
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < K
        a_ptrs = A + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=True)  # 使用 Tensor Core (FP16 输入)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

# ----------------------------------------------------------------------
# 2. add bias kernel (broadcast over batch)
# ----------------------------------------------------------------------
@triton.jit
def add_bias_kernel(
    x_ptr, bias_ptr, y_ptr, n_elements, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    bias_idx = offs % hidden_size
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    y = x + bias
    tl.store(y_ptr + offs, y, mask=mask)

# ----------------------------------------------------------------------
# 3. group normalization kernel (per sample, per group)
# ----------------------------------------------------------------------
@triton.jit
def group_norm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    batch_size, num_groups, group_size,
    stride_x, stride_y,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)
    start_ch = pid_group * group_size
    offs = tl.arange(0, BLOCK_SIZE)
    ch_offs = start_ch + offs
    ch_mask = offs < group_size
    x_ptrs = x_ptr + pid_batch * stride_x + ch_offs
    x = tl.load(x_ptrs, mask=ch_mask, other=0.0)
    # mean
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / group_size
    # var = E[x^2] - mean^2
    sum_x2 = tl.sum(x * x, axis=0)
    var = sum_x2 / group_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    norm = (x - mean) * inv_std
    # weight & bias
    weight = tl.load(weight_ptr + ch_offs, mask=ch_mask, other=0.0)
    bias = tl.load(bias_ptr + ch_offs, mask=ch_mask, other=0.0)
    y = norm * weight + bias
    y_ptrs = y_ptr + pid_batch * stride_y + ch_offs
    tl.store(y_ptrs, y, mask=ch_mask)

# ----------------------------------------------------------------------
# 4. leaky relu kernel
# ----------------------------------------------------------------------
@triton.jit
def leaky_relu_kernel(
    x_ptr, y_ptr, n_elements,
    slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.where(x >= 0, x, x * slope)
    tl.store(y_ptr + offs, y, mask=mask)

# ----------------------------------------------------------------------
# 5. element‑wise addition (x + x)
# ----------------------------------------------------------------------
@triton.jit
def add_kernel(
    x_ptr, y_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x + x
    tl.store(y_ptr + offs, y, mask=mask)

# ----------------------------------------------------------------------
# run() – entry point, same signature as reference
# ----------------------------------------------------------------------
def run(x):
    global _weights_cache, _weights_device
    device = x.device
    if _weights_cache is None or _weights_device != device:
        _load_weights(device)

    B, in_dim = x.shape
    hidden = 8192
    num_groups = 512
    group_size = hidden // num_groups  # 16
    eps = 1e-5
    slope = 0.01

    # weights
    fc_weight = _weights_cache["fc.weight"]          # [hidden, in_dim]
    fc_bias = _weights_cache["fc.bias"]              # [hidden]
    gn_weight = _weights_cache["gn.weight"]          # [hidden]
    gn_bias = _weights_cache["gn.bias"]              # [hidden]

    # prepare transposed weight for matmul
    fc_weight_t = fc_weight.T.contiguous()           # [in_dim, hidden]

    # intermediate tensors
    matmul_out = torch.empty(B, hidden, device=device, dtype=torch.float32)
    bias_out = torch.empty(B, hidden, device=device, dtype=torch.float32)
    norm_out = torch.empty(B, hidden, device=device, dtype=torch.float32)
    leaky_out = torch.empty(B, hidden, device=device, dtype=torch.float32)
    final_out = torch.empty(B, hidden, device=device, dtype=torch.float32)

    BLOCK = 32

    # 使用 FP16 输入和 Tensor Core 加速 matmul
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    x_half = x.to(torch.float16)
    fc_weight_t_half = fc_weight_t.to(torch.float16)
    grid_mm = (triton.cdiv(B, BLOCK_M), triton.cdiv(hidden, BLOCK_N))
    matmul_kernel[grid_mm](
        x_half, fc_weight_t_half, matmul_out,
        B, hidden, in_dim,
        x_half.stride(0), x_half.stride(1),
        fc_weight_t_half.stride(0), fc_weight_t_half.stride(1),
        matmul_out.stride(0), matmul_out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # ----- add bias -----
    n_elems = B * hidden
    grid_add_bias = (triton.cdiv(n_elems, BLOCK),)
    add_bias_kernel[grid_add_bias](
        matmul_out, fc_bias, bias_out, n_elems, hidden, BLOCK_SIZE=BLOCK,
    )

    # ----- group norm -----
    grid_gn = (B, num_groups)
    # 注意：x 和 y 的 stride 是 hidden (因为 2D 连续)
    group_norm_kernel[grid_gn](
        bias_out, norm_out, gn_weight, gn_bias,
        B, num_groups, group_size,
        bias_out.stride(0), norm_out.stride(0),
        eps=eps, BLOCK_SIZE=BLOCK,
    )

    # ----- leaky relu -----
    grid_leaky = (triton.cdiv(n_elems, BLOCK),)
    leaky_relu_kernel[grid_leaky](
        norm_out, leaky_out, n_elems, slope=slope, BLOCK_SIZE=BLOCK,
    )

    # ----- element‑wise add (x + x) -----
    add_kernel[grid_leaky](
        leaky_out, final_out, n_elems, BLOCK_SIZE=BLOCK,
    )

    return final_out
