import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    coeff = 0.044715
    sqrt2pi = math.sqrt(2.0 / math.pi)  # ≈ 0.79788456

    x3 = x * x * x
    inner = x + coeff * x3
    tmp = sqrt2pi * inner

    # tanh via sigmoid: tanh(z) = 2 * sigmoid(2*z) - 1
    tanh_val = 2.0 * tl.sigmoid(2.0 * tmp) - 1.0

    y = 0.5 * x * (1.0 + tanh_val)

    tl.store(output_ptr + offsets, y, mask=mask)

def run(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, output, n_elements)
    return output
