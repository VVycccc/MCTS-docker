import torch
import triton
import triton.language as tl

_weights_path = "problems/kb_level1/40_layernorm_weights.pt"
_w_cache = None


@triton.jit
def _layernorm_partial_kernel(
    x_ptr, partial_ptr,
    N,
    BLOCK: tl.constexpr,
):
    # Each program reduces one chunk of one row; atomically accumulate
    # per-row sum and sum-of-squares into partial_ptr[row*2 : row*2+2].
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(x)
    sum_x2 = tl.sum(x * x)
    tl.atomic_add(partial_ptr + row * 2, sum_x)
    tl.atomic_add(partial_ptr + row * 2 + 1, sum_x2)


@triton.jit
def _layernorm_apply_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, partial_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    # Each program normalizes one chunk of one row using the per-row
    # statistics produced by _layernorm_partial_kernel.
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    sum_x = tl.load(partial_ptr + row * 2)
    sum_x2 = tl.load(partial_ptr + row * 2 + 1)
    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * w + b
    tl.store(y_ptr + row * N + offs, y, mask=mask)


def run(x):
    global _w_cache
    if _w_cache is None or str(_w_cache[0].device) != str(x.device):
        sd = torch.load(_weights_path, map_location='cpu', weights_only=True)
        w = sd['ln.weight'].to(x.device).contiguous()
        b = sd['ln.bias'].to(x.device).contiguous()
        _w_cache = (w, b)
    w, b = _w_cache

    x = x.contiguous()
    y = torch.empty_like(x)

    N = w.numel()              # normalized_shape flattened: features * dim1 * dim2
    rows = x.numel() // N      # batch dim
    eps = 1e-5                 # nn.LayerNorm default

    BLOCK = 4096
    num_chunks = triton.cdiv(N, BLOCK)
    partial = torch.zeros((rows, 2), device=x.device, dtype=torch.float32)
    grid = (rows, num_chunks)
    _layernorm_partial_kernel[grid](x, partial, N, BLOCK=BLOCK, num_warps=8)
    _layernorm_apply_kernel[grid](x, w, b, y, partial, N, eps, BLOCK=BLOCK, num_warps=8)
    return y
