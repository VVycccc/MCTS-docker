import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/52_Conv2d_Activation_BatchNorm_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    raw = torch.load(_weights_path, map_location='cpu', weights_only=True)
    _weights = {k: v.to(device).contiguous() for k, v in raw.items()}
    # Pre-transpose conv weight to [C_in*9, C_out] for coalesced GEMM access
    conv_weight = _weights['conv.weight']  # [C_out, C_in, 3, 3]
    C_out, C_in = conv_weight.shape[0], conv_weight.shape[1]
    _weights['conv_weight_t'] = conv_weight.view(C_out, C_in * 9).t().contiguous()  # [C_in*9, C_out]
    _device = str(device)


@triton.jit
def fused_conv_act_bn_kernel(
    x_ptr, w_t_ptr, b_ptr, bn_w_ptr, bn_b_ptr, rm_ptr, rv_ptr, out_ptr,
    N, C_in, C_out, H, W, H_out, W_out, K_dim,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    M = N * H_out * W_out
    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decompose output pixel index: m -> (n_batch, ho, wo)
    hwout = H_out * W_out
    n_idx = offs_m // hwout
    rem = offs_m % hwout
    ho = rem // W_out
    wo = rem % W_out

    # GEMM accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # im2col + GEMM loop over K = C_in * 9
    for k_start in range(0, K_dim, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]

        # Decompose k index: k -> (ci, kh, kw)
        ci = k_idx // 9
        kh = (k_idx % 9) // 3
        kw = k_idx % 3

        # Compute input indices for im2col on-the-fly
        ih = ho[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = wo[:, None] + kw[None, :]  # [BLOCK_M, BLOCK_K]
        x_idx = n_idx[:, None] * (C_in * H * W) + ci[None, :] * (H * W) + ih * W + iw

        # Load input tile (im2col) and weight tile (pre-transposed [K, N])
        a = tl.load(x_ptr + x_idx, mask=mask_m[:, None], other=0.0)
        b = tl.load(w_t_ptr + k_idx[:, None] * C_out + offs_n[None, :],
                     mask=mask_n[None, :], other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    # Add conv bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Fused activation: mish(x) = x * tanh(softplus(x))
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    act = th * acc

    # Fused batchnorm: (act - rm) / sqrt(rv + eps) * bn_w + bn_b
    rm = tl.load(rm_ptr + offs_n, mask=mask_n, other=0.0)
    rv = tl.load(rv_ptr + offs_n, mask=mask_n, other=0.0)
    bn_w = tl.load(bn_w_ptr + offs_n, mask=mask_n, other=0.0)
    bn_b = tl.load(bn_b_ptr + offs_n, mask=mask_n, other=0.0)
    bn_out = (act - rm[None, :]) / tl.sqrt(rv[None, :] + eps) * bn_w[None, :] + bn_b[None, :]

    # Store output in [N, C_out, H_out, W_out] (NCHW) layout
    out_idx = n_idx[:, None] * (C_out * hwout) + offs_n[None, :] * hwout + ho[:, None] * W_out + wo[:, None]
    tl.store(out_ptr + out_idx, bn_out, mask=mask_m[:, None] & mask_n[None, :])


def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    x = x.contiguous()
    N, C_in, H, W = x.shape
    conv_weight_t = _weights['conv_weight_t']  # [C_in*9, C_out]
    conv_bias = _weights['conv.bias']
    bn_weight = _weights['bn.weight']
    bn_bias = _weights['bn.bias']
    bn_running_mean = _weights['bn.running_mean']
    bn_running_var = _weights['bn.running_var']

    C_out = conv_weight_t.shape[1]
    K_dim = conv_weight_t.shape[0]  # C_in * 9
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
        x, conv_weight_t, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var, out,
        N, C_in, C_out, H, W, H_out, W_out, K_dim,
        eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=2,
    )

    return out
