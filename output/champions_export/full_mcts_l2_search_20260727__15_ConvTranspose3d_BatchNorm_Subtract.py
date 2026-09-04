import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/15_ConvTranspose3d_BatchNorm_Subtract_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, out_ptr,
    N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
    stride, padding,
    K: tl.constexpr, C_IN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    spatial_total = D_out * H_out * W_out
    mask = offs < spatial_total

    ow = offs % W_out
    tmp = offs // W_out
    oh = tmp % H_out
    od = tmp // H_out

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for kd in range(K):
        for kh in range(K):
            for kw in range(K):
                id_num = od + padding - kd
                ih_num = oh + padding - kh
                iw_num = ow + padding - kw
                valid = (id_num % stride == 0) & (ih_num % stride == 0) & (iw_num % stride == 0)
                id_val = id_num // stride
                ih_val = ih_num // stride
                iw_val = iw_num // stride
                valid = valid & (id_val >= 0) & (id_val < D_in) & (ih_val >= 0) & (ih_val < H_in) & (iw_val >= 0) & (iw_val < W_in)

                id_val = tl.where(valid, id_val, 0)
                ih_val = tl.where(valid, ih_val, 0)
                iw_val = tl.where(valid, iw_val, 0)
                load_mask = mask & valid

                for ic in range(C_IN):
                    w_idx = ((ic * C_out + pid_c) * K + kd) * K * K + kh * K + kw
                    w_val = tl.load(w_ptr + w_idx).to(tl.float32)

                    x_idx = ((pid_n * C_IN + ic) * D_in + id_val) * H_in * W_in + ih_val * W_in + iw_val
                    x_val = tl.load(x_ptr + x_idx, mask=load_mask, other=0.0)

                    acc += w_val * x_val.to(tl.float32)

    b_val = tl.load(b_ptr + pid_c).to(tl.float32)
    scale = tl.load(scale_ptr + pid_c).to(tl.float32)
    acc = (acc + b_val) * scale

    out_offs = (pid_n * C_out + pid_c) * (D_out * H_out * W_out) + od * (H_out * W_out) + oh * W_out + ow
    tl.store(out_ptr + out_offs, acc, mask=mask)

def run(x):
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    x = x.contiguous()

    conv_weight = _weights['conv_transpose.weight']
    conv_bias = _weights['conv_transpose.bias']
    bn_weight = _weights['batch_norm.weight']
    bn_bias = _weights['batch_norm.bias']
    bn_running_mean = _weights['batch_norm.running_mean']
    bn_running_var = _weights['batch_norm.running_var']

    N, C_in, D_in, H_in, W_in = x.shape
    C_out = conv_weight.shape[1]
    K = conv_weight.shape[2]
    stride = 2
    padding = 1

    D_out = (D_in - 1) * stride - 2 * padding + K
    H_out = (H_in - 1) * stride - 2 * padding + K
    W_out = (W_in - 1) * stride - 2 * padding + K

    # BN: y = x * scale + shift, scale = weight/std, shift = bias - running_mean * scale
    # 最终结果 = y - mean(y, spatial) = scale * (x - mean(x, spatial)), shift 项被消除
    eps = 1e-5
    bn_scale = (bn_weight / torch.sqrt(bn_running_var + eps)).contiguous()

    conv_out = torch.empty(N, C_out, D_out, H_out, W_out, device=x.device, dtype=torch.float32)
    spatial_total = D_out * H_out * W_out
    BLOCK = 256
    grid = (N, C_out, triton.cdiv(spatial_total, BLOCK))
    conv_transpose3d_kernel[grid](
        x, conv_weight, conv_bias, bn_scale, conv_out,
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        stride, padding,
        K=K, C_IN=C_in,
        BLOCK=BLOCK,
    )

    spatial_mean = conv_out.mean(dim=(2, 3, 4), keepdim=True)
    conv_out = conv_out - spatial_mean

    return conv_out
