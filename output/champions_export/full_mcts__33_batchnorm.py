import torch
import torch.nn as nn
import triton
import triton.language as tl

_weights_path = "problems/kb_level1/33_batchnorm_weights.pt"
_bn_cache = None
_bn_device = None

@triton.jit
def batchnorm_kernel(
    x_ptr, y_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, HW, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_block = tl.program_id(1)

    c = pid_bc % C

    local_offs = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = pid_bc * HW + local_offs
    mask = local_offs < HW

    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = weight * inv_std
    shift = bias - mean * scale

    x = tl.load(x_ptr + offs, mask=mask)
    y = x * scale + shift

    tl.store(y_ptr + offs, y, mask=mask)

def run(x):
    global _bn_cache, _bn_device
    if _bn_cache is None or _bn_device != str(x.device):
        state_dict = torch.load(_weights_path, map_location=x.device, weights_only=True)
        weight = state_dict['bn.weight']
        bias = state_dict['bn.bias']
        running_mean = state_dict['bn.running_mean']
        running_var = state_dict['bn.running_var']
        _bn_cache = (weight, bias, running_mean, running_var)
        _bn_device = str(x.device)

    weight, bias, running_mean, running_var = _bn_cache

    y = torch.empty_like(x)
    N = x.numel()
    B, C, H, W = x.shape
    HW = H * W
    eps = 1e-5
    BLOCK_SIZE = 16384

    num_blocks_per_channel = triton.cdiv(HW, BLOCK_SIZE)
    grid = (B * C, num_blocks_per_channel)

    batchnorm_kernel[grid](
        x, y,
        running_mean, running_var, weight, bias,
        N, C, HW, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return y
