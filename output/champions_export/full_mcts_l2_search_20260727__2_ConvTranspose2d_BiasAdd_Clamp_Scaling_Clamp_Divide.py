import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.bias
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x * self.scaling_factor
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x / self.scaling_factor
        return x

import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.jit
def conv_transpose2d_seed_kernel(
    x_ptr, weight_ptr, conv_bias_ptr, bias_ptr, output_ptr,
    B, IC: tl.constexpr, OC, H, W, OH, OW,
    K: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
    SCALING,
    OH_OW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_oc = tl.program_id(2)

    s_offs = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    s_mask = s_offs < OH_OW
    oy = s_offs // OW
    ox = s_offs % OW

    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    oc_mask = oc_offs < OC

    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    acc += cb[:, None]

    HW = H * W
    OCKK = OC * K * K
    KK = K * K

    for ky in range(K):
        for kx in range(K):
            iy_raw = oy + P - ky
            ix_raw = ox + P - kx
            cond = (iy_raw % S == 0) & (ix_raw % S == 0)
            iy = iy_raw // S
            ix = ix_raw // S
            in_bounds = cond & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)

            for ic_start in range(0, IC, BLOCK_K):
                ic_offs = ic_start + tl.arange(0, BLOCK_K)
                ic_mask = ic_offs < IC

                x_idx = pid_b * (IC * HW) + ic_offs[:, None] * HW + iy[None, :] * W + ix[None, :]
                x_val = tl.load(x_ptr + x_idx, mask=ic_mask[:, None] & in_bounds[None, :], other=0.0)

                w_idx = oc_offs[:, None] * KK + ic_offs[None, :] * OCKK + ky * K + kx
                w_val = tl.load(weight_ptr + w_idx, mask=oc_mask[:, None] & ic_mask[None, :], other=0.0)

                acc += tl.dot(w_val, x_val, allow_tf32=True)

    b_val = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc += b_val[:, None]

    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 1.0)
    acc = acc * SCALING
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 1.0)
    acc = acc / SCALING

    out_idx = pid_b * (OC * OH_OW) + oc_offs[:, None] * OH_OW + s_offs[None, :]
    tl.store(output_ptr + out_idx, acc, mask=oc_mask[:, None] & s_mask[None, :])

def run(x):
    global _weights, _device
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)

    bias = _weights['bias']
    conv_transpose_bias = _weights['conv_transpose.bias']
    conv_transpose_weight = _weights['conv_transpose.weight']

    B, IC, H, W = x.shape
    OC = conv_transpose_weight.shape[1]
    K = conv_transpose_weight.shape[2]
    S = 2
    P = 1
    OP = 1
    SCALING = 2.0

    OH = (H - 1) * S - 2 * P + K + OP
    OW = (W - 1) * S - 2 * P + K + OP

    output = torch.empty(B, OC, OH, OW, device=x.device, dtype=torch.float32)

    OH_OW = OH * OW
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (B, triton.cdiv(OH_OW, BLOCK_M), triton.cdiv(OC, BLOCK_N))

    conv_transpose2d_seed_kernel[grid](
        x, conv_transpose_weight, conv_transpose_bias, bias, output,
        B, IC, OC, H, W, OH, OW,
        K, S, P,
        SCALING,
        OH_OW,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return output
