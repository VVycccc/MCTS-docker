import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 112
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

_weights_path = "problems/kb_level1/35_groupnorm_weights.pt"
_W = None
_W_device = None


@triton.jit
def groupnorm_kernel(x_ptr, y_ptr, gamma_ptr, beta_ptr,
                     B, C, eps,
                     G: tl.constexpr, CPG: tl.constexpr, D1D2: tl.constexpr,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // G
    g = pid % G

    total = CPG * D1D2
    base = b * C * D1D2 + g * total

    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, total, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + idx)
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)

    mean = sum_x / total
    var = sum_x2 / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c in tl.static_range(0, CPG):
        c_global = g * CPG + c
        gamma = tl.load(gamma_ptr + c_global)
        beta = tl.load(beta_ptr + c_global)
        channel_base = base + c * D1D2
        for off in range(0, D1D2, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + channel_base + idx)
            y = (x - mean) * rstd * gamma + beta
            tl.store(y_ptr + channel_base + idx, y)


def run(x):
    global _W, _W_device
    if _W is None or _W_device != x.device:
        sd = torch.load(_weights_path, map_location='cpu', weights_only=True)
        gamma = sd['gn.weight'].to(x.device)
        beta = sd['gn.bias'].to(x.device)
        _W = (gamma, beta)
        _W_device = x.device
    gamma, beta = _W

    B, C, D1, D2 = x.shape
    G = num_groups
    CPG = C // G
    D1D2 = D1 * D2
    eps = 1e-5

    y = torch.empty_like(x)
    grid = (B * G,)
    groupnorm_kernel[grid](x, y, gamma, beta,
                           B, C, eps,
                           G=G, CPG=CPG, D1D2=D1D2,
                           BLOCK=1024, num_warps=4, num_stages=3)
    return y
