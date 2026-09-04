import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_swish_bias_kernel(
    x_ptr, w_ptr, b_linear_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wm, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_block_ptr = tl.make_block_ptr(
            base=x_ptr, shape=(M, K), strides=(stride_xm, stride_xk),
            offsets=(pid_m * BLOCK_M, k),
            block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=w_ptr, shape=(K, N), strides=(stride_wn, stride_wm),
            offsets=(k, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N), order=(0, 1)
        )
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b, allow_tf32=True)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    b_linear = tl.load(b_linear_ptr + n_offsets, mask=n_mask, other=0.0)
    bias_val = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)

    accumulator += b_linear[None, :]

    result = tl.sigmoid(accumulator) * accumulator
    result = result + bias_val[None, :]

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr, shape=(M, N), strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    tl.store(output_block_ptr, result, boundary_check=(0, 1))


@triton.jit
def group_norm_kernel(
    x_ptr, output_ptr, weight_ptr, bias_ptr,
    stride_xm, stride_xn,
    stride_om, stride_on,
    M, N, eps,
    NUM_GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride_xm
    output_ptr += row * stride_om

    g_offsets = tl.arange(0, NUM_GROUPS)
    c_offsets = tl.arange(0, CHANNELS_PER_GROUP)
    cols = g_offsets[:, None] * CHANNELS_PER_GROUP + c_offsets[None, :]

    x = tl.load(x_ptr + cols * stride_xn).to(tl.float32)

    mean = tl.sum(x, axis=1) / CHANNELS_PER_GROUP
    x_centered = x - mean[:, None]
    var = tl.sum(x_centered * x_centered, axis=1) / CHANNELS_PER_GROUP
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + cols)
    b = tl.load(bias_ptr + cols)

    x_hat = (x - mean[:, None]) * rstd[:, None]
    y = x_hat * w + b
    tl.store(output_ptr + cols * stride_on, y)


batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/37_Matmul_Swish_Sum_GroupNorm_weights.pt"
_weights = None

def run(x, *args):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _weights = {k: v.to(x.device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

    w = _weights['matmul.weight']
    b_linear = _weights['matmul.bias']
    bias = _weights['bias']
    gn_weight = _weights['group_norm.weight']
    gn_bias = _weights['group_norm.bias']

    M, K = x.shape
    N = w.shape[0]
    num_groups = 64
    channels_per_group = N // num_groups

    intermediate = torch.empty((M, N), dtype=x.dtype, device=x.device)
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fused_matmul_swish_bias_kernel[grid](
        x, w, b_linear, bias, intermediate,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        intermediate.stride(0), intermediate.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_stages=3, num_warps=8,
    )

    group_norm_kernel[(M,)](
        intermediate, output, gn_weight, gn_bias,
        intermediate.stride(0), intermediate.stride(1),
        output.stride(0), output.stride(1),
        M, N, 1e-5,
        NUM_GROUPS=num_groups,
        CHANNELS_PER_GROUP=channels_per_group,
        num_warps=8,
    )

    return output
