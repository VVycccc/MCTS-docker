import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                  N, C_in, D, H, W, C_out, D_out, H_out, W_out,
                  KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # im2col + matmul 实现 Conv3d
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    K = C_in * KD * KH * KW
    S = D_out * H_out * W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # 分解 k -> (c_in, kd, kh, kw)
        k_cin = offs_k // (KD * KH * KW)
        k_rem = offs_k % (KD * KH * KW)
        k_kd = k_rem // (KH * KW)
        k_rem2 = k_rem % (KH * KW)
        k_kh = k_rem2 // KW
        k_kw = k_rem2 % KW

        # 权重指针 (BLOCK_M, BLOCK_K)
        w_ptrs = (w_ptr + offs_m[:, None] * (C_in * KD * KH * KW)
                  + k_cin[None, :] * (KD * KH * KW)
                  + k_kd[None, :] * (KH * KW)
                  + k_kh[None, :] * KW
                  + k_kw[None, :])
        w_mask = (offs_m[:, None] < C_out) & (offs_k[None, :] < K)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # 分解 n -> (d_out, h_out, w_out)
        n_dout = offs_n // (H_out * W_out)
        n_rem = offs_n % (H_out * W_out)
        n_hout = n_rem // W_out
        n_wout = n_rem % W_out

        # 输入位置
        d_in = n_dout[None, :] + k_kd[:, None]
        h_in = n_hout[None, :] + k_kh[:, None]
        w_in = n_wout[None, :] + k_kw[:, None]

        # 输入指针 (BLOCK_K, BLOCK_N)
        x_ptrs = (x_ptr + pid_b * (C_in * D * H * W)
                  + k_cin[:, None] * (D * H * W)
                  + d_in * (H * W)
                  + h_in * W
                  + w_in)
        x_mask = ((offs_k[:, None] < K) & (offs_n[None, :] < S)
                  & (d_in < D) & (h_in < H) & (w_in < W))
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc = tl.dot(w_block, x_block, acc=acc, allow_tf32=True)

    # 加偏置
    b_ptrs = b_ptr + offs_m
    b_vals = tl.load(b_ptrs, mask=offs_m < C_out, other=0.0)
    acc += b_vals[:, None]

    # 存储结果
    n_dout = offs_n // (H_out * W_out)
    n_rem = offs_n % (H_out * W_out)
    n_hout = n_rem // W_out
    n_wout = n_rem % W_out

    out_ptrs = (out_ptr + pid_b * (C_out * D_out * H_out * W_out)
                + offs_m[:, None] * (D_out * H_out * W_out)
                + n_dout[None, :] * (H_out * W_out)
                + n_hout[None, :] * W_out
                + n_wout[None, :])
    out_mask = (offs_m[:, None] < C_out) & (offs_n[None, :] < S)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def min_softmax_fused_kernel(in_ptr, out_ptr,
                             N, C, D, H_out, W_out,
                             BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr):
    # 融合 min-along-D + softmax-along-C
    # 每块处理 BLOCK_C 通道 × BLOCK_S 空间位置
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    # 分解空间位置 -> (h, w)
    s_h = offs_s // W_out
    s_w = offs_s % W_out

    c_mask = offs_c < C
    s_mask = offs_s < (H_out * W_out)

    # 初始化 min 为 inf
    min_vals = tl.full((BLOCK_C, BLOCK_S), float('inf'), dtype=tl.float32)

    # 沿 D 维度循环累加 min
    for d in range(0, D):
        ptrs = (in_ptr + pid_n * (C * D * H_out * W_out)
                + offs_c[:, None] * (D * H_out * W_out)
                + d * (H_out * W_out)
                + s_h[None, :] * W_out
                + s_w[None, :])
        mask = c_mask[:, None] & s_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=float('inf'))
        min_vals = tl.minimum(min_vals, vals)

    # Softmax 沿 channel 维度 (axis=0)
    valid = c_mask[:, None] & s_mask[None, :]
    # 将无效位置设为 -inf 以保证 max 计算正确
    min_vals = tl.where(valid, min_vals, -float('inf'))

    max_val = tl.max(min_vals, axis=0)  # (BLOCK_S,)
    shifted = min_vals - max_val[None, :]
    exp_vals = tl.exp(shifted)
    exp_vals = tl.where(valid, exp_vals, 0.0)
    sum_val = tl.sum(exp_vals, axis=0)  # (BLOCK_S,)
    out = exp_vals / sum_val[None, :]

    # 存储结果
    out_ptrs = (out_ptr + pid_n * (C * H_out * W_out)
                + offs_c[:, None] * (H_out * W_out)
                + s_h[None, :] * W_out
                + s_w[None, :])
    tl.store(out_ptrs, out, mask=valid)


batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/24_Conv3d_Min_Softmax_weights.pt"
_weights = None
def run(x, *args):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _weights = {k: v.to(x.device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

    w = _weights['conv.weight']
    b = _weights['conv.bias']

    N, C_in, D_val, H_val, W_val = x.shape
    C_out = out_channels
    KD = KH = KW = kernel_size
    D_out = D_val - KD + 1
    H_out = H_val - KH + 1
    W_out = W_val - KW + 1

    # Conv3d
    conv_out = torch.empty((N, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 128, 32
    grid_conv = (triton.cdiv(C_out, BLOCK_M), triton.cdiv(D_out * H_out * W_out, BLOCK_N), N)
    conv3d_kernel[grid_conv](
        x, w, b, conv_out,
        N, C_in, D_val, H_val, W_val, C_out, D_out, H_out, W_out,
        KD, KH, KW,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_stages=3, num_warps=8
    )

    # 融合 min-along-D + softmax-along-C
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    BLOCK_C = 32
    BLOCK_S = 128
    grid_fused = (N, triton.cdiv(H_out * W_out, BLOCK_S))
    min_softmax_fused_kernel[grid_fused](
        conv_out, out,
        N, C_out, D_out, H_out, W_out,
        BLOCK_C, BLOCK_S,
        num_warps=4
    )

    return out
