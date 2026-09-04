import torch
import triton
import triton.language as tl

# 全局缓存
_conv_weight = None
_conv_bias = None
_conv_device = None

@triton.jit
def im2col_kernel(
    input_ptr, col_ptr,
    N, C, H_in, W_in, H_out, W_out,
    kernel_size, stride, padding,
    M, K,
    BLOCK_SIZE: tl.constexpr
):
    """将 NCHW 输入展开为 [M, K] 行主序矩阵，M = N*H_out*W_out, K = C*kernel_size*kernel_size"""
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = M * K
    mask = pid < total

    # pid 平坦索引 → m, k
    m = pid // K
    k = pid - m * K

    # 从 m 分解出 n, h_out, w_out
    out_hw = H_out * W_out
    n = m // out_hw
    rem = m - n * out_hw
    h_out = rem // W_out
    w_out = rem - h_out * W_out

    # 从 k 分解出 ci, kh, kw
    k_size2 = kernel_size * kernel_size
    ci = k // k_size2
    rem_k = k - ci * k_size2
    kh = rem_k // kernel_size
    kw = rem_k - kh * kernel_size

    # 输入窗口位置
    h_in = h_out * stride - padding + kh
    w_in = w_out * stride - padding + kw

    valid = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in) & mask

    # 输入索引 (NCHW)
    input_idx = n * (C * H_in * W_in) + ci * (H_in * W_in) + h_in * W_in + w_in
    input_val = tl.load(input_ptr + input_idx, mask=valid, other=0.0)

    # 写入 col 矩阵
    col_ptr += pid
    tl.store(col_ptr, input_val, mask=mask)

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    """标准 Triton matmul，使用 tensor core 和共享内存 tile"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    offs_am = start_m + tl.arange(0, BLOCK_M)
    offs_bn = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_am[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_bn[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    offs_cm = start_m + tl.arange(0, BLOCK_M)
    offs_cn = start_n + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * N + offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

def run(x):
    global _conv_weight, _conv_bias, _conv_device
    if _conv_weight is None or _conv_device != str(x.device):
        state_dict = torch.load(
            "problems/kb_level1/50_conv_standard_2d_square_input_square_kernel_weights.pt",
            map_location='cpu',
            weights_only=True
        )
        _conv_weight = state_dict['conv1.weight'].to(x.device).contiguous()
        _conv_bias = state_dict['conv1.bias'].to(x.device).contiguous()
        _conv_device = str(x.device)

    N, C_in, H_in, W_in = x.shape
    kernel_size = 11
    stride = 4
    padding = 2
    C_out = _conv_weight.shape[0]

    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1

    M = N * H_out * W_out
    K = C_in * kernel_size * kernel_size

    # 1. im2col
    col = torch.empty(M, K, device=x.device, dtype=torch.float16)
    grid_im2col = (triton.cdiv(M * K, 512),)
    im2col_kernel[grid_im2col](
        x, col,
        N, C_in, H_in, W_in, H_out, W_out,
        kernel_size, stride, padding,
        M, K,
        BLOCK_SIZE=512
    )

    # 2. 权重转置为 [K, C_out]
    weight_T = _conv_weight.reshape(C_out, K).T.contiguous().half()  # [K, C_out]

    # 3. matmul: col [M, K] × weight_T [K, C_out] → out [M, C_out]
    out_mat = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
    grid_matmul = (triton.cdiv(M, 128) * triton.cdiv(C_out, 128),)
    matmul_kernel[grid_matmul](
        col, weight_T, out_mat,
        M, C_out, K,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        GROUP_M=8
    )

    # 4. 加 bias 并 reshape
    output = out_mat + _conv_bias[None, :]  # broadcast
    output = output.view(N, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()

    return output
