import torch
import torch.nn as nn
import triton
import triton.language as tl


# 错误分析：shared memory不足 (Required: 147456, Hardware limit: 101376)
# 原因：BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, num_stages=4 配置过大
# 解决方案：减小block size和num_stages，确保shared memory使用量在限制内
# 计算公式：shared_mem ≈ num_stages * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 4 bytes
# 目标：shared_mem < 101376 bytes


@triton.autotune(
    configs=[
        # shared: 3 * (128*32 + 32*128) * 4 = 3 * 8192 * 4 = 98304 ✓
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        # shared: 4 * (64*32 + 32*128) * 4 = 4 * 6144 * 4 = 98304 ✓
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        # shared: 4 * (128*32 + 32*64) * 4 = 4 * 6144 * 4 = 98304 ✓
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        # shared: 3 * (64*64 + 64*64) * 4 = 3 * 8192 * 4 = 98304 ✓
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        # shared: 4 * (64*32 + 32*64) * 4 = 4 * 4096 * 4 = 65536 ✓
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        # shared: 3 * (128*32 + 32*64) * 4 = 3 * 6144 * 4 = 73728 ✓
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        # shared: 3 * (64*32 + 32*128) * 4 = 3 * 6144 * 4 = 73728 ✓
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
    restore_value=['output_ptr'],
)
@triton.jit
def akg_l2_9_Matmul_Subtract_Multiply_ReLU_it6_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    subtract_value,
    multiply_value,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_block_ptr = tl.make_block_ptr(
            base=x_ptr, shape=(M, K), strides=(stride_xm, stride_xk),
            offsets=(pid_m * BLOCK_SIZE_M, k),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=weight_ptr, shape=(K, N), strides=(stride_wk, stride_wn),
            offsets=(k, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
        )
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator = tl.dot(a, b, acc=accumulator)

    # 加上bias
    bias_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_mask = bias_offsets < N
    bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
    accumulator += bias[None, :]

    # 减法
    accumulator = accumulator - subtract_value
    # 乘法
    accumulator = accumulator * multiply_value
    # ReLU
    accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    # 存储结果
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr, shape=(M, N), strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(output_block_ptr, accumulator, boundary_check=(0, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        torch.manual_seed(0)
        linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(linear.weight.clone())
        self.bias = nn.Parameter(linear.bias.clone())
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        M, K = x.shape
        N = self.weight.shape[0]
        output = torch.empty((M, N), dtype=x.dtype, device=x.device)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N']),
        )

        akg_l2_9_Matmul_Subtract_Multiply_ReLU_it6_kernel[grid](
            x, self.weight, self.bias, output,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1),
            self.subtract_value,
            self.multiply_value,
        )

        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x)
        x = x - self.subtract_value
        x = x * self.multiply_value
        x = torch.relu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level2/9_Matmul_Subtract_Multiply_ReLU_weights.pt"
_MODEL = None
def run(x, *args):
    global _MODEL
    if _MODEL is None:
        import torch
        torch.manual_seed(0)
        _MODEL = ModelNew(*get_init_inputs())
        try:
            _ref = Model(*get_init_inputs())
            _ref.load_state_dict(torch.load(_weights_path, map_location='cpu', weights_only=True))
            _rp = list(_ref.parameters()); _np = list(_MODEL.parameters())
            for _pn, _pr in zip(_np, _rp):
                if _pn.shape == _pr.shape:
                    _pn.data.copy_(_pr.data)
        except FileNotFoundError:
            pass  # 权重文件缺失：保留 ModelNew 的 seed 初始化（与 AKG 内部验证条件一致）
        _MODEL = _MODEL.to(x.device).eval()
    return _MODEL(x, *args)
