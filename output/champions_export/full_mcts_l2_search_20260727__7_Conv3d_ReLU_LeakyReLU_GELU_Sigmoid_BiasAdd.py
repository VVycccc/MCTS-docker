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
    IC: tl.constexpr, K: tl.constexpr, RK: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SPATIAL: tl.constexpr, BLOCK_RK: tl.constexpr,
):
    pid = tl.program_id(0)

    num_spatial = OD * OH * OW
    num_pid_oc = tl.cdiv(OC, BLOCK_OC)
    num_pid_spatial = tl.cdiv(num_spatial, BLOCK_SPATIAL)

    n = pid // (num_pid_oc * num_pid_spatial)
    rem = pid % (num_pid_oc * num_pid_spatial)
    pid_oc = rem // num_pid_spatial
    pid_spatial = rem % num_pid_spatial

    # Output channel offsets
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Spatial position offsets
    sp_offs = pid_spatial * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    sp_mask = sp_offs < num_spatial

    # Decompose spatial position into (od, oh, ow)
    od = sp_offs // (OH * OW)
    rem_sp = sp_offs % (OH * OW)
    oh = rem_sp // OW
    ow = rem_sp % OW

    # Precompute spatial_base (independent of loop variable r)
    spatial_base = n * (IC * ID * IH * IW) + od * (IH * IW) + oh * IW + ow

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SPATIAL, BLOCK_OC), dtype=tl.float32)

    # Iterate over the reduced dimension (ic * K * K * K)
    for r_start in range(0, RK, BLOCK_RK):
        r_offs = r_start + tl.arange(0, BLOCK_RK)
        r_mask = r_offs < RK

        # Decompose r into (ic, kd, kh, kw)
        ic = r_offs // (K * K * K)
        rem_r = r_offs % (K * K * K)
        kd = rem_r // (K * K)
        rem_r2 = rem_r % (K * K)
        kh = rem_r2 // K
        kw = rem_r2 % K

        # Compute r_base as 1D vector (cheap scalar ops)
        r_base = ic * (ID * IH * IW) + kd * (IH * IW) + kh * IW + kw

        # Compute input indices: [BLOCK_SPATIAL, BLOCK_RK] — single broadcast add
        in_idx = spatial_base[:, None] + r_base[None, :]

        # Compute weight indices: [BLOCK_RK, BLOCK_OC]
        w_idx = r_offs[:, None] + oc_offs[None, :] * RK

        # Load input and weight tiles
        x_tile = tl.load(x_ptr + in_idx, mask=sp_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)
        w_tile = tl.load(w_ptr + w_idx, mask=r_mask[:, None] & oc_mask[None, :], other=0.0).to(tl.float32)

        # Matrix multiply via tensor cores
        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=True)

    # Add conv bias
    conv_bias = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc = acc + conv_bias[None, :]

    # Apply activations: ReLU -> LeakyReLU -> GELU -> Sigmoid
    acc = tl.maximum(acc, 0.0)
    acc = tl.where(acc >= 0.0, acc, 0.01 * acc)
    acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # Add bias
    bias_val = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc = acc + bias_val[None, :]

    # Store output
    out_idx = n * (OC * num_spatial) + oc_offs[None, :] * num_spatial + sp_offs[:, None]
    tl.store(out_ptr + out_idx, acc, mask=sp_mask[:, None] & oc_mask[None, :])

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
    BLOCK_RK = 32
    RK = IC * K * K * K

    num_spatial = OD * OH * OW
    grid = (N * triton.cdiv(OC, BLOCK_OC) * triton.cdiv(num_spatial, BLOCK_SPATIAL),)

    fused_kernel[grid](
        x, conv_weight, conv_bias, bias, out,
        N, OC, OD, OH, OW, ID, IH, IW,
        IC=IC, K=K, RK=RK,
        BLOCK_OC=BLOCK_OC, BLOCK_SPATIAL=BLOCK_SPATIAL, BLOCK_RK=BLOCK_RK,
        num_warps=8, num_stages=3,
    )
    return out
