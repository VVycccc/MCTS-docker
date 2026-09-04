import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level1/40_layernorm_weights.pt"
_W_cache = None
_W_device = None

@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride_b, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)
    base = pid * stride_b + pid_c * N

    # Single-pass Online Welford: stream over N, accumulate count/mean/M2,
    # then normalize and write back in the same loop.
    count = 0.0
    mean = 0.0
    M2 = 0.0
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + base + idx)
        n = BLOCK_SIZE
        count += n
        delta = x - mean
        mean += tl.sum(delta) / count
        delta2 = x - mean
        M2 += tl.sum(delta * delta2)

    var = M2 / N
    inv_std = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + base + idx)
        w = tl.load(w_ptr + idx)
        b = tl.load(b_ptr + idx)
        y = (x - mean) * inv_std * w + b
        tl.store(out_ptr + base + idx, y)


def run(x):
    global _W_cache, _W_device

    if _W_cache is None or _W_device != str(x.device):
        state_dict = torch.load(_weights_path, map_location='cpu', weights_only=True)
        weight = state_dict['ln.weight'].to(x.device).contiguous()
        bias = state_dict['ln.bias'].to(x.device).contiguous()
        _W_cache = (weight, bias)
        _W_device = str(x.device)

    weight, bias = _W_cache

    x = x.contiguous()
    batch_size = x.shape[0]
    N = x.shape[1] * x.shape[2] * x.shape[3]

    out = torch.empty_like(x)

    x_flat = x.reshape(-1)
    out_flat = out.reshape(-1)
    weight_flat = weight.reshape(-1)
    bias_flat = bias.reshape(-1)

    num_channels = x.shape[1]
    norm_size = N // num_channels

    grid = (batch_size, num_channels)
    layernorm_kernel[grid](
        x_flat, weight_flat, bias_flat, out_flat,
        N, norm_size, 1e-5,
        BLOCK_SIZE=2048, num_warps=8, num_stages=4,
    )

    return out
