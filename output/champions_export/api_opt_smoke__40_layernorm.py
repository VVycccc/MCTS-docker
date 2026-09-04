import torch
import triton
import triton.language as tl

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level1/40_layernorm_weights.pt"
_weights_cache = None
_weights_device = None

@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # Single pass: compute mean and variance via E[x] and E[x^2]
    sum_val = 0.0
    sum_sq_val = 0.0
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + pid * N + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq_val += tl.sum(x * x)

    mean = sum_val / N
    var = sum_sq_val / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and write
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + pid * N + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std * w + b
        tl.store(y_ptr + pid * N + offs, y, mask=mask)


def run(x):
    global _weights_cache, _weights_device
    if _weights_cache is None or _weights_device != str(x.device):
        state_dict = torch.load(_weights_path, map_location='cpu', weights_only=True)
        weight = state_dict['ln.weight'].to(x.device).contiguous()
        bias = state_dict['ln.bias'].to(x.device).contiguous()
        _weights_cache = (weight, bias)
        _weights_device = str(x.device)

    weight, bias = _weights_cache

    x_contiguous = x.contiguous()
    y = torch.empty_like(x_contiguous)

    batch_size = x_contiguous.shape[0]
    features = x_contiguous.shape[1]
    N = x_contiguous.shape[2] * x_contiguous.shape[3]  # 256 * 256 = 65536

    grid = (batch_size * features,)  # 16 * 64 = 1024 programs
    layernorm_kernel[grid](
        x_contiguous, weight, bias, y,
        N, 1e-5,
        BLOCK=2048,
    )
    return y
