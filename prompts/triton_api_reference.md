# Triton DSL Quick Reference

## Kernel Definition

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(ptr_a, ptr_b, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(ptr_a + offs, mask=mask)
    y = x * 2.0
    tl.store(ptr_b + offs, y, mask=mask)

def run(a):
    b = torch.empty_like(a)
    grid = (triton.cdiv(a.numel(), BLOCK_SIZE),)
    my_kernel[grid](a, b, a.numel(), BLOCK_SIZE=1024)
    return b
```

## @triton.autotune (USE SPARINGLY)

Only use when tile sizes genuinely need tuning. **Maximum 3 configs.**

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel(...):
    ...

def run(a, b):
    ...
    kernel[grid](a, b, c, M, N, K, ...)
    return c
```

- Auto-tuned params (BLOCK_M etc.) must NOT be passed as keyword args to kernel call
- Shared memory limit: ~ BLOCK_M × BLOCK_K × dtype_size × num_stages × 2 + BLOCK_K × BLOCK_N × dtype_size × num_stages × 2 < 99 KB

## Memory Operations

```python
# Load (unmasked)
x = tl.load(ptr + offs)

# Load (masked)
x = tl.load(ptr + offs, mask=mask, other=0.0)

# Store
tl.store(ptr + offs, value, mask=mask)

# Atomic add
tl.atomic_add(ptr + offs, value, mask=mask)
```

## Compute

```python
# Matrix multiply (tensor core)
c = tl.dot(a, b, acc=c, allow_tf32=True)    # FP32 accum + TF32 multiply
c = tl.dot(a, b, acc=c, allow_tf32=False)   # Full FP32 on CUDA cores (slower)

# Elementwise
y = tl.exp(x)
y = tl.log(x)
y = tl.sigmoid(x)
y = tl.maximum(a, b)
y = tl.where(mask, true_val, false_val)

# IMPORTANT: tl.tanh / tl.softplus / tl.mish DO NOT EXIST in this Triton version.
# Implement them via the available primitives:
y = 2.0 * tl.sigmoid(2.0 * x) - 1.0              # tanh(x)
sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))  # softplus(x)  (numerically stable)
y = x * (2.0 * tl.sigmoid(2.0 * sp) - 1.0)       # mish(x) = x * tanh(softplus(x))
y = x * tl.where(x >= 3.0, 1.0, tl.where(x <= -3.0, 0.0, (x + 3.0) / 6.0))  # hardswish(x)
y = tl.where(x >= 3.0, 1.0, tl.where(x <= -3.0, 0.0, (x + 3.0) / 6.0))  # hardsigmoid(x)
gelu = x * 0.5 * (1.0 + 2.0 * tl.sigmoid(2.0 * (x / 1.41421356)))  # gelu(x) via erf approx (use tl.erf if available)

# Reduction
s = tl.sum(x, axis=0)

# Cast
y = tl.cast(x, tl.float32)
```

## Indexing and Constants

```python
# Range (start and end MUST be powers of 2)
idx = tl.arange(0, 128)

# Program ID / grid
pid = tl.program_id(0)  # 0=block_x, 1=block_y, 2=block_z
num_pid = tl.num_programs(0)

# Ceiling division (for grid size)
grid = (triton.cdiv(N, BLOCK_SIZE),)

# tl.constexpr values MUST be kernel parameters, NOT computed from runtime tensors
```

## Common Patterns

**EVEN_K fast path (no masking when K divisible by BLOCK_K):**
```python
@triton.jit
def kernel(..., BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr):
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
        else:
            k_rem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k < k_rem, other=0.0)
```

**GROUP_M swizzle (L2 cache locality):**
```python
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

## Hardware Constraints (RTX 3090 Ampere SM 8.6)

| Limit | Value |
|-------|-------|
| Max shared memory per block | 99 KB (101376 bytes) |
| Max threads per block | 1024 (num_warps × 32) |
| Max registers per thread | 255 |
| Max blocks per SM | 32 |

- `num_warps=4` → 128 threads, `num_warps=8` → 256 threads, `num_warps=16` → 512 threads
- `num_stages=2` → lower shared memory, `num_stages=3-4` → better latency hiding
- BLOCK sizes must be ≥ 16 and powers of 2
- triton.arange(0, N) requires N to be a power of 2
