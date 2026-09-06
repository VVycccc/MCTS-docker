import torch
import triton
import triton.language as tl

_W = None
_W_device = None

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=1),
    ],
    key=['D'],
)
@triton.jit
def layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr, D, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * D

    # Pass 1: compute mean and variance simultaneously (E[x^2] - E[x]^2)
    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, D, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + idx)
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)
    mean = sum_x / D
    var = sum_x2 / D - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and write
    for off in range(0, D, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + idx)
        gamma = tl.load(gamma_ptr + idx)
        beta = tl.load(beta_ptr + idx)
        y = (x - mean) * rstd * gamma + beta
        tl.store(out_ptr + base + idx, y)


def run(x):
    global _W, _W_device
    if _W is None or _W_device != str(x.device):
        sd = torch.load("problems/kb_level1/40_layernorm_weights.pt", map_location='cpu', weights_only=True)
        gamma = sd['ln.weight'].to(x.device).contiguous()
        beta = sd['ln.bias'].to(x.device).contiguous()
        _W = (gamma, beta)
        _W_device = str(x.device)
    gamma, beta = _W
    out = torch.empty_like(x)
    B = x.shape[0]
    D = x.shape[1] * x.shape[2] * x.shape[3]
    eps = 1e-5
    grid = (B,)
    layernorm_kernel[grid](x, gamma, beta, out, D, eps)
    return out