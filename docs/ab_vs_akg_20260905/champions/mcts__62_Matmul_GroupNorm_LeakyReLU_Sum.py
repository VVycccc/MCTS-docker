import torch
import torch.nn as nn
import triton
import triton.language as tl

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.fc(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        x = x + x
        return x


batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]
# Frozen weights (loaded from .pt at module init):
#   fc.weight: [8192, 8192] (float32)
#   fc.bias: [8192] (float32)
#   gn.weight: [8192] (float32)
#   gn.bias: [8192] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/62_Matmul_GroupNorm_LeakyReLU_Sum_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    fc_bias = _weights['fc.bias']  # [8192]
    fc_weight = _weights['fc.weight']  # [8192, 8192]
    gn_bias = _weights['gn.bias']  # [8192]
    gn_weight = _weights['gn.weight']  # [8192]

    x = _torch.nn.functional.linear(x, fc_weight, fc_bias)
    x = _torch.nn.functional.group_norm(x, num_groups=1, weight=gn_weight, bias=gn_bias)
    x = _torch.nn.functional.leaky_relu(x)
    x = x + x
    return x

# --- UNIVERSAL RUN (验证用，调 Model.forward) ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_matmul_gn_lrelu_2x_kernel(
    x_ptr, w_ptr, fc_b_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    GROUP_M: tl.constexpr = 8
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
            w = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            k_rem = K - k * BLOCK_K
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_rem), other=0.0)
        acc = tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16), acc=acc, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add fc bias
    fc_b = tl.load(fc_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += fc_b[None, :]

    # GroupNorm: reshape (BLOCK_M, BLOCK_N) -> (BLOCK_M * NUM_GROUPS_PER_TILE, GROUP_SIZE)
    NUM_GROUPS_PER_TILE: tl.constexpr = BLOCK_N // GROUP_SIZE
    FLAT_ROWS: tl.constexpr = BLOCK_M * NUM_GROUPS_PER_TILE
    acc_flat = tl.reshape(acc, (FLAT_ROWS, GROUP_SIZE))
    mean = tl.sum(acc_flat, axis=1) / GROUP_SIZE
    diff = acc_flat - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + EPS)
    acc_norm_flat = diff * rstd[:, None]
    acc_norm = tl.reshape(acc_norm_flat, (BLOCK_M, BLOCK_N))

    # Apply gn weight and bias
    gn_w = tl.load(gn_w_ptr + offs_n, mask=offs_n < N, other=0.0)
    gn_b = tl.load(gn_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc_norm = acc_norm * gn_w[None, :] + gn_b[None, :]

    # LeakyReLU
    acc_norm = tl.where(acc_norm >= 0, acc_norm, acc_norm * NEG_SLOPE)

    # 2x (x + x)
    acc_norm = acc_norm + acc_norm

    # Store
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc_norm, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


_fc_weight_bf16 = None

def run(x):
    global _weights, _device, _fc_weight_bf16
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
        _fc_weight_bf16 = None

    if _fc_weight_bf16 is None:
        _fc_weight_bf16 = _weights['fc.weight'].to(torch.bfloat16)
    fc_weight = _fc_weight_bf16  # [8192, 8192] bf16
    fc_bias = _weights['fc.bias']      # [8192]
    gn_weight = _weights['gn.weight']  # [8192]
    gn_bias = _weights['gn.bias']      # [8192]

    M, K = x.shape
    N = fc_weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    GROUP_SIZE = hidden_size // num_groups  # 16

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _fused_matmul_gn_lrelu_2x_kernel[grid](
        x, fc_weight, fc_bias, gn_weight, gn_bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        fc_weight.stride(0), fc_weight.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=GROUP_SIZE,
        EPS=1e-5,
        NEG_SLOPE=0.01,
        EVEN_K=True,
    )

    return out