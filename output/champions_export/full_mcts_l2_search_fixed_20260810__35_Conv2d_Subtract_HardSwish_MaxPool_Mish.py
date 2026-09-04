import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/35_Conv2d_Subtract_HardSwish_MaxPool_Mish_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def fused_conv_sub_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    subtract_value,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    HW_out: tl.constexpr,
    K_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_pid_m = tl.cdiv(out_channels, BLOCK_M)
    num_pid_n = tl.cdiv(HW_out, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K_dim, BLOCK_K)):
        k_vals = k_start * BLOCK_K + offs_k  # (BLOCK_K,)

        # Load A tile (weight reshaped): (BLOCK_M, BLOCK_K)
        a_ptrs = w_ptr + offs_m[:, None] * K_dim + k_vals[None, :]
        if EVEN_K:
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=offs_m[:, None] < out_channels, other=0.0)
        else:
            a_mask = (offs_m[:, None] < out_channels) & (k_vals[None, :] < K_dim)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile (im2col on-the-fly): (BLOCK_K, BLOCK_N)
        ic = k_vals // 9               # (BLOCK_K,)
        kh = (k_vals % 9) // 3        # (BLOCK_K,)
        kw = (k_vals % 9) % 3         # (BLOCK_K,)

        oh = offs_n // W_out           # (BLOCK_N,)
        ow = offs_n % W_out           # (BLOCK_N,)

        ih = oh[None, :] + kh[:, None]  # (BLOCK_K, BLOCK_N)
        iw = ow[None, :] + kw[:, None]  # (BLOCK_K, BLOCK_N)

        x_idx = pid_b * in_channels * H * W + ic[:, None] * H * W + ih * W + iw  # (BLOCK_K, BLOCK_N)

        b_mask_n = offs_n[None, :] < HW_out  # (1, BLOCK_N)
        if EVEN_K:
            b = tl.load(x_ptr + x_idx, mask=b_mask_n, other=0.0)
        else:
            b_mask = b_mask_n & (k_vals[:, None] < K_dim)
            b = tl.load(x_ptr + x_idx, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    # Fused epilogue: bias + subtract + hardswish
    if EVEN_M:
        b_val = tl.load(b_ptr + offs_m)
    else:
        b_val = tl.load(b_ptr + offs_m, mask=offs_m < out_channels, other=0.0)
    acc += b_val[:, None]
    acc = acc - subtract_value
    acc = acc * tl.where(acc >= 3.0, 1.0, tl.where(acc <= -3.0, 0.0, (acc + 3.0) / 6.0))

    # Store
    out_idx = pid_b * out_channels * HW_out + offs_m[:, None] * HW_out + offs_n[None, :]
    if EVEN_M:
        out_mask = offs_n[None, :] < HW_out
    else:
        out_mask = (offs_m[:, None] < out_channels) & (offs_n[None, :] < HW_out)
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def maxpool2d_kernel(x_ptr, out_ptr, N, C, H, W, H_out, W_out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid_w = tl.program_id(1)

    num_c_h = C * H_out
    pid_n = pid // num_c_h
    remainder = pid % num_c_h
    pid_c = remainder // H_out
    pid_oh = remainder % H_out

    offs_w = pid_w * BLOCK + tl.arange(0, BLOCK)
    mask_w = offs_w < W_out

    ih0 = pid_oh * 2
    iw0 = offs_w * 2

    base = pid_n * C * H * W + pid_c * H * W + ih0 * W
    v00 = tl.load(x_ptr + base + iw0, mask=mask_w, other=-1.0e30)
    v01 = tl.load(x_ptr + base + iw0 + 1, mask=mask_w, other=-1.0e30)
    v10 = tl.load(x_ptr + base + W + iw0, mask=mask_w, other=-1.0e30)
    v11 = tl.load(x_ptr + base + W + iw0 + 1, mask=mask_w, other=-1.0e30)

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    out_idx = pid_n * C * H_out * W_out + pid_c * H_out * W_out + pid_oh * W_out + offs_w
    tl.store(out_ptr + out_idx, result, mask=mask_w)


@triton.jit
def mish_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    result = x * tanh_sp
    tl.store(out_ptr + offs, result, mask=mask)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    conv_weight = _weights['conv.weight']
    conv_bias = _weights['conv.bias']

    batch_size, in_channels, H, W = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = 3
    H_out = H - kernel_size + 1
    W_out = W - kernel_size + 1
    HW_out = H_out * W_out
    K_dim = in_channels * 9

    x = x.contiguous()

    # Reshape weight for GEMM: (out_channels, in_channels*9)
    conv_weight_gemm = conv_weight.reshape(out_channels, K_dim).contiguous()

    # Step 1: Fused Conv2d + Subtract + HardSwish (GEMM with tl.dot tensor core)
    hs_out = torch.empty(batch_size, out_channels, H_out, W_out, device=x.device, dtype=torch.float32)
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    num_pid_m = triton.cdiv(out_channels, BLOCK_M)
    num_pid_n = triton.cdiv(HW_out, BLOCK_N)
    grid_conv = (num_pid_m * num_pid_n, batch_size)
    EVEN_K = (K_dim % BLOCK_K == 0)
    EVEN_M = (out_channels % BLOCK_M == 0)
    fused_conv_sub_hardswish_kernel[grid_conv](
        x, conv_weight_gemm, conv_bias, hs_out,
        0.5,
        in_channels=in_channels, out_channels=out_channels, H=H, W=W,
        H_out=H_out, W_out=W_out, HW_out=HW_out, K_dim=K_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K, EVEN_M=EVEN_M,
        num_stages=3, num_warps=4,
    )

    # Step 2: MaxPool2d (kernel_size=2, stride=2)
    pool_H_out = H_out // 2
    pool_W_out = W_out // 2
    pool_out = torch.empty(batch_size, out_channels, pool_H_out, pool_W_out, device=x.device, dtype=torch.float32)
    BLOCK_P = 256
    grid_pool = (batch_size * out_channels * pool_H_out, triton.cdiv(pool_W_out, BLOCK_P))
    maxpool2d_kernel[grid_pool](
        hs_out, pool_out,
        batch_size, out_channels, H_out, W_out, pool_H_out, pool_W_out,
        BLOCK=BLOCK_P,
    )

    # Step 3: Mish
    n2 = pool_out.numel()
    out = torch.empty_like(pool_out)
    BLOCK_MISH = 1024
    grid2 = (triton.cdiv(n2, BLOCK_MISH),)
    mish_kernel[grid2](pool_out, out, n2, BLOCK=BLOCK_MISH, num_warps=4)

    return out
