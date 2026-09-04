import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    raw = torch.load(_weights_path, map_location='cpu', weights_only=True)
    _weights = {k: v.to(device).contiguous() for k, v in raw.items()}
    _device = str(device)


@triton.jit
def conv_transpose_bn_tanh_kernel(
    x_ptr, w_mat_ptr, b_ptr, out_ptr,
    N, C_IN, C_OUT, H, W, H_OUT, W_OUT, HW_OUT, PADDING,
    KSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_n = N * HW_OUT
    n_valid = offs_n < total_n

    n_idx = offs_n // HW_OUT
    hw = offs_n % HW_OUT
    oh = hw // W_OUT
    ow = hw % W_OUT

    K_DIM = C_IN * KSIZE * KSIZE
    K_HW = KSIZE * KSIZE
    HW_in = H * W
    CHW_in = C_IN * HW_in

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_DIM, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_DIM

        w_ptrs = w_mat_ptr + offs_m[:, None] * K_DIM + offs_k[None, :]
        w_tile = tl.load(w_ptrs, mask=(offs_m[:, None] < C_OUT) & k_mask[None, :], other=0.0)

        c_in_k = offs_k[:, None] // K_HW
        kh_k = (offs_k[:, None] % K_HW) // KSIZE
        kw_k = offs_k[:, None] % KSIZE

        ih = oh[None, :] + PADDING - kh_k
        iw = ow[None, :] + PADDING - kw_k
        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

        x_idx = n_idx[None, :] * CHW_in + c_in_k * HW_in + ih * W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=valid & n_valid[None, :] & k_mask[:, None], other=0.0)

        acc = tl.dot(w_tile, x_tile, acc=acc, allow_tf32=True)

    b_val = tl.load(b_ptr + offs_m, mask=offs_m < C_OUT, other=0.0)
    acc += b_val[:, None]

    acc = 2.0 * tl.sigmoid(2.0 * acc) - 1.0

    out_idx = n_idx[None, :] * (C_OUT * HW_OUT) + offs_m[:, None] * HW_OUT + hw[None, :]
    out_mask = (offs_m[:, None] < C_OUT) & n_valid[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def bn_kernel(
    x_ptr, out_ptr, w_ptr, b_ptr, rm_ptr, rv_ptr,
    N, C, H, W,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * H * W
    mask = offs < total
    HW = H * W
    CHW = C * HW
    c = (offs % CHW) // HW

    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0)
    rm = tl.load(rm_ptr + c, mask=mask, other=0.0)
    rv = tl.load(rv_ptr + c, mask=mask, other=0.0)
    w = tl.load(w_ptr + c, mask=mask, other=0.0)
    b = tl.load(b_ptr + c, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(rv + 1e-5)
    out = (x_val - rm) * inv_std * w + b
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def tanh_kernel(x_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def maxpool_kernel(
    x_ptr, out_ptr,
    N, C, H, W, H_OUT, W_OUT,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * H_OUT * W_OUT
    mask = offs < total
    HW_out = H_OUT * W_OUT
    CHW_out = C * HW_out
    n = offs // CHW_out
    rem = offs % CHW_out
    c = rem // HW_out
    rem2 = rem % HW_out
    oh = rem2 // W_OUT
    ow = rem2 % W_OUT

    ih0 = 2 * oh
    iw0 = 2 * ow
    HW_in = H * W
    CHW_in = C * HW_in
    base = n * CHW_in + c * HW_in

    a = tl.load(x_ptr + base + ih0 * W + iw0, mask=mask, other=0.0)
    b = tl.load(x_ptr + base + ih0 * W + iw0 + 1, mask=mask, other=0.0)
    c2 = tl.load(x_ptr + base + (ih0 + 1) * W + iw0, mask=mask, other=0.0)
    d = tl.load(x_ptr + base + (ih0 + 1) * W + iw0 + 1, mask=mask, other=0.0)
    m = tl.maximum(tl.maximum(a, b), tl.maximum(c2, d))
    tl.store(out_ptr + offs, m, mask=mask)


@triton.jit
def gn_kernel(
    x_ptr, out_ptr, w_ptr, b_ptr,
    N, C, H, W, NUM_GROUPS, CPG,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    HW = H * W
    spatial = CPG * HW
    base = n * C * HW + g * CPG * HW

    num_iters = (spatial + BLOCK - 1) // BLOCK
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(0, num_iters):
        off = i * BLOCK
        idxs = off + tl.arange(0, BLOCK)
        m = idxs < spatial
        x = tl.load(x_ptr + base + idxs, mask=m, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / spatial
    var = sum_sq / spatial - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    for i in range(0, num_iters):
        off = i * BLOCK
        idxs = off + tl.arange(0, BLOCK)
        m = idxs < spatial
        x = tl.load(x_ptr + base + idxs, mask=m, other=0.0)
        c_local = idxs // HW
        c = g * CPG + c_local
        w = tl.load(w_ptr + c, mask=m, other=0.0)
        b = tl.load(b_ptr + c, mask=m, other=0.0)
        out = (x - mean) * inv_std * w + b
        tl.store(out_ptr + base + idxs, out, mask=m)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    x = x.contiguous()
    device = x.device
    N, C_IN, H, W = x.shape
    C_OUT = 128
    KSIZE = 5
    PADDING = 1
    STRIDE = 1
    NUM_GROUPS = 8
    BLOCK = 64

    H_OUT = (H - 1) * STRIDE - 2 * PADDING + KSIZE
    W_OUT = (W - 1) * STRIDE - 2 * PADDING + KSIZE

    conv_w = _weights['conv_transpose.weight']
    conv_b = _weights['conv_transpose.bias']
    bn_w = _weights['batch_norm.weight']
    bn_b = _weights['batch_norm.bias']
    bn_rm = _weights['batch_norm.running_mean']
    bn_rv = _weights['batch_norm.running_var']
    gn_w = _weights['group_norm.weight']
    gn_b = _weights['group_norm.bias']

    # Stage 1-3: ConvTranspose2d + BatchNorm(fused into weights) + Tanh (fused epilogue) via im2col+GEMM
    eps = 1e-5
    bn_scale = bn_w / torch.sqrt(bn_rv + eps)
    bn_shift = bn_b - bn_rm * bn_scale
    fused_w = conv_w * bn_scale.view(1, C_OUT, 1, 1)
    fused_b = conv_b * bn_scale + bn_shift
    w_mat = fused_w.permute(1, 0, 2, 3).reshape(C_OUT, C_IN * KSIZE * KSIZE).contiguous()

    y3 = torch.empty(N, C_OUT, H_OUT, W_OUT, device=device, dtype=torch.float32)
    HW_OUT = H_OUT * W_OUT
    total_n = N * HW_OUT
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid_gemm = (triton.cdiv(C_OUT, BLOCK_M), triton.cdiv(total_n, BLOCK_N))
    conv_transpose_bn_tanh_kernel[grid_gemm](
        x, w_mat, fused_b, y3,
        N, C_IN, C_OUT, H, W, H_OUT, W_OUT, HW_OUT, PADDING,
        KSIZE=KSIZE, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=3, num_warps=8,
    )

    # Stage 4: MaxPool2d (kernel=2, stride=2)
    H_P = H_OUT // 2
    W_P = W_OUT // 2
    y4 = torch.empty(N, C_OUT, H_P, W_P, device=device, dtype=torch.float32)
    total4 = N * C_OUT * H_P * W_P
    grid4 = (triton.cdiv(total4, BLOCK),)
    maxpool_kernel[grid4](y3, y4, N, C_OUT, H_OUT, W_OUT, H_P, W_P, BLOCK=BLOCK)

    # Stage 5: GroupNorm (num_groups=8)
    y5 = torch.empty_like(y4)
    CPG = C_OUT // NUM_GROUPS
    grid5 = (N * NUM_GROUPS,)
    gn_kernel[grid5](y4, y5, gn_w, gn_b, N, C_OUT, H_P, W_P, NUM_GROUPS, CPG, BLOCK=BLOCK)

    return y5
