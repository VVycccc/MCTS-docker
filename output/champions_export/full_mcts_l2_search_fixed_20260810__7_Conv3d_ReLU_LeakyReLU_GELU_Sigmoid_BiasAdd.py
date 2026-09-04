import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device).contiguous() for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
    _device = str(device)

@triton.jit
def fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, OC, OD, OH, OW, ID, IH, IW,
    IC: tl.constexpr, K: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SPATIAL: tl.constexpr, BLOCK_KK: tl.constexpr,
):
    pid = tl.program_id(0)

    spatial = OD * OH * OW
    num_pid_oc = tl.cdiv(OC, BLOCK_OC)
    num_pid_spatial = tl.cdiv(spatial, BLOCK_SPATIAL)

    n = pid // (num_pid_oc * num_pid_spatial)
    rem = pid % (num_pid_oc * num_pid_spatial)
    oc_block = rem // num_pid_spatial
    spatial_block = rem % num_pid_spatial

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    spatial_offs = spatial_block * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)

    oc_mask = oc_offs < OC
    spatial_mask = spatial_offs < spatial

    # Decode spatial offsets into (od, oh, ow)
    od = spatial_offs // (OH * OW)
    rem_s = spatial_offs % (OH * OW)
    oh = rem_s // OW
    ow = rem_s % OW

    # Initialize accumulator with conv_bias, broadcast to 2D
    conv_bias_vals = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc = conv_bias_vals[:, None] + tl.zeros((BLOCK_OC, BLOCK_SPATIAL), dtype=tl.float32)

    # Iterate over IC*K*K*K in tiles of BLOCK_KK
    IC_KKK = IC * K * K * K
    for kk in range(0, IC_KKK, BLOCK_KK):
        kk_offs = kk + tl.arange(0, BLOCK_KK)
        kk_mask = kk_offs < IC_KKK

        # Decode kk_offs into (ic, kd, kh, kw)
        ic = kk_offs // (K * K * K)
        rem_kk = kk_offs % (K * K * K)
        kd = rem_kk // (K * K)
        rem_kk2 = rem_kk % (K * K)
        kh = rem_kk2 // K
        kw = rem_kk2 % K

        # Load weights: w[oc, ic, kd, kh, kw] -> shape (BLOCK_OC, BLOCK_KK)
        w_offs = oc_offs[:, None] * (IC * K * K * K) + kk_offs[None, :]
        w_mask = oc_mask[:, None] & kk_mask[None, :]
        w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0).to(tl.float32)

        # Load input: x[n, ic, od+kd, oh+kh, ow+kw] -> shape (BLOCK_KK, BLOCK_SPATIAL)
        in_offs = (
            n * (IC * ID * IH * IW)
            + ic[:, None] * (ID * IH * IW)
            + (od[None, :] + kd[:, None]) * (IH * IW)
            + (oh[None, :] + kh[:, None]) * IW
            + (ow[None, :] + kw[:, None])
        )
        in_mask = kk_mask[:, None] & spatial_mask[None, :]
        x_vals = tl.load(x_ptr + in_offs, mask=in_mask, other=0.0).to(tl.float32)

        # GEMM via tensor core
        acc = tl.dot(w_vals, x_vals, acc=acc, allow_tf32=True)

    # Apply activations: ReLU (subsumes LeakyReLU) -> GELU -> Sigmoid -> BiasAdd
    acc = tl.maximum(acc, 0.0)
    acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # Add per-channel bias
    bias_val = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc = acc + bias_val[:, None]

    # Store output
    out_offs = n * (OC * spatial) + oc_offs[:, None] * spatial + spatial_offs[None, :]
    out_mask = oc_mask[:, None] & spatial_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)

def run(x):
    if _weights is None or _device != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']
    conv_bias = _weights['conv.bias']
    conv_weight = _weights['conv.weight']

    x = x.contiguous()
    N, IC, ID, IH, IW = x.shape
    OC = conv_weight.shape[0]
    K = conv_weight.shape[2]
    OD = ID - K + 1
    OH = IH - K + 1
    OW = IW - K + 1

    out = torch.empty(N, OC, OD, OH, OW, device=x.device, dtype=torch.float32)

    BLOCK_OC = 32
    BLOCK_SPATIAL = 128
    BLOCK_KK = 32

    spatial = OD * OH * OW
    grid = (N * triton.cdiv(OC, BLOCK_OC) * triton.cdiv(spatial, BLOCK_SPATIAL),)

    fused_kernel[grid](
        x, conv_weight, conv_bias, bias, out,
        N, OC, OD, OH, OW, ID, IH, IW,
        IC=IC, K=K,
        BLOCK_OC=BLOCK_OC, BLOCK_SPATIAL=BLOCK_SPATIAL, BLOCK_KK=BLOCK_KK,
        num_warps=4, num_stages=2,
    )
    return out
