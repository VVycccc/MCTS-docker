import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/52_Conv2d_Activation_BatchNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)


@triton.jit
def fused_conv_act_bn_kernel(
    x_ptr, w_ptr, b_ptr, bn_w_ptr, bn_b_ptr, rm_ptr, rv_ptr, out_ptr,
    N, C_in, C_out, H, W, H_out, W_out, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M = N * H_out * W_out
    K = C_in * 9

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    hwout = H_out * W_out
    n_batch = offs_m // hwout
    spatial = offs_m % hwout
    ho = spatial // W_out
    wo = spatial % W_out

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        ci = offs_k // 9
        khkw = offs_k % 9
        kh = khkw // 3
        kw = khkw % 3

        ih = ho[:, None] + kh[None, :]
        iw = wo[:, None] + kw[None, :]
        x_idx = n_batch[:, None] * (C_in * H * W) + ci[None, :] * (H * W) + ih * W + iw
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        w_idx = offs_n[None, :] * K + offs_k[:, None]
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_val = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_val, w_val, acc=acc, allow_tf32=True)

    bias_val = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_val[None, :]

    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    act = th * acc

    rm = tl.load(rm_ptr + offs_n, mask=mask_n, other=0.0)
    rv = tl.load(rv_ptr + offs_n, mask=mask_n, other=0.0)
    bn_w = tl.load(bn_w_ptr + offs_n, mask=mask_n, other=0.0)
    bn_b = tl.load(bn_b_ptr + offs_n, mask=mask_n, other=0.0)

    bn_out = (act - rm[None, :]) / tl.sqrt(rv[None, :] + eps) * bn_w[None, :] + bn_b[None, :]

    out_idx = n_batch[:, None] * (C_out * hwout) + offs_n[None, :] * hwout + spatial[:, None]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_idx, bn_out, mask=out_mask)


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    x = x.contiguous()
    N, C_in, H, W = x.shape
    conv_weight = _weights['conv.weight']
    conv_bias = _weights['conv.bias']
    bn_weight = _weights['bn.weight']
    bn_bias = _weights['bn.bias']
    bn_running_mean = _weights['bn.running_mean']
    bn_running_var = _weights['bn.running_var']

    C_out = conv_weight.shape[0]
    H_out = H - 2
    W_out = W - 2
    eps = 1e-5

    M = N * H_out * W_out
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

    out = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    fused_conv_act_bn_kernel[grid](
        x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var, out,
        N, C_in, C_out, H, W, H_out, W_out, eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=2, num_warps=4,
    )

    return out
