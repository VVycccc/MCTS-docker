import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def accel_l2_29_Matmul_Mish_Mish_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, k),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(k, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
        )
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=True)

    bias_off = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias = tl.load(bias_ptr + bias_off, mask=bias_off < N, other=0.0)
    accumulator += bias[None, :]

    # Mish 1: x * tanh(softplus(x))
    abs_acc = tl.abs(accumulator)
    sp = tl.maximum(accumulator, 0.0) + tl.log(1.0 + tl.exp(-abs_acc))
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    accumulator = accumulator * tanh_sp

    # Mish 2
    abs_acc = tl.abs(accumulator)
    sp = tl.maximum(accumulator, 0.0) + tl.log(1.0 + tl.exp(-abs_acc))
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    accumulator = accumulator * tanh_sp

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


batch_size = 1024
in_features = 8192
out_features = 8192

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/29_Matmul_Mish_Mish_weights.pt"
_weights = None

def run(x, *args):
    global _weights
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _weights = {k: v.to(x.device) for k, v in torch.load(_weights_path, map_location='cpu', weights_only=True).items()}
        _weights['linear.weight_fp16'] = _weights['linear.weight'].to(torch.float16)
    weight = _weights['linear.weight']
    bias = _weights['linear.bias']
    weight_fp16 = _weights['linear.weight_fp16']
    x_fp16 = x.to(torch.float16)
    M, K = x.shape
    N = weight.shape[0]
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    accel_l2_29_Matmul_Mish_Mish_kernel[grid](
        x_fp16, weight_fp16, output, bias,
        M, N, K,
        x_fp16.stride(0), x_fp16.stride(1),
        1, weight_fp16.stride(0),
        output.stride(0), output.stride(1),
    )
    return output
