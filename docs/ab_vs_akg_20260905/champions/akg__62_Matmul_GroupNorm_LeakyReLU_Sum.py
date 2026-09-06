import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    矩阵乘法内核: C = A @ B + bias
    A: (M, K), B: (K, N) [实际存储为(N, K)的转置], C: (M, N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 沿K轴迭代
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
        accumulator += tl.dot(a, b)
    
    # 加上bias
    bias_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_mask = bias_offs < N
    bias = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0)
    accumulator += bias[None, :]
    
    # 存储结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


@triton.jit
def groupnorm_leakyrelu_sum_kernel(
    input_ptr, output_ptr, weight_ptr, bias_ptr,
    n_rows, n_cols, num_groups,
    stride_row,
    eps,
    negative_slope,
    BLOCK_SIZE_ROW: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
):
    """
    GroupNorm + LeakyReLU + Sum (x + x) 融合内核
    每个程序处理 BLOCK_SIZE_ROW 行的一个 group
    """
    pid_row = tl.program_id(0)
    pid_group = tl.program_id(1)
    
    # 计算行偏移
    row_start = pid_row * BLOCK_SIZE_ROW
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_ROW)
    row_mask = row_offsets < n_rows
    
    # 计算列偏移（当前group的通道范围）
    col_start = pid_group * CHANNELS_PER_GROUP
    col_offsets = col_start + tl.arange(0, CHANNELS_PER_GROUP)
    
    # 加载数据: (BLOCK_SIZE_ROW, CHANNELS_PER_GROUP)
    row_ptrs = input_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :]
    mask = row_mask[:, None]
    data = tl.load(row_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # GroupNorm: 计算均值和方差（沿channel维度）
    mean = tl.sum(data, axis=1) / CHANNELS_PER_GROUP
    var = tl.sum((data - mean[:, None]) * (data - mean[:, None]), axis=1) / CHANNELS_PER_GROUP
    
    # 归一化
    rstd = 1 / tl.sqrt(var + eps)
    x_hat = (data - mean[:, None]) * rstd[:, None]
    
    # 加载weight和bias
    w = tl.load(weight_ptr + col_offsets)
    b = tl.load(bias_ptr + col_offsets)
    
    # 仿射变换
    gn_output = x_hat * w[None, :] + b[None, :]
    
    # LeakyReLU
    leaky_output = tl.where(gn_output >= 0, gn_output, gn_output * negative_slope)
    
    # Sum (x + x = 2x)
    output = leaky_output + leaky_output
    
    # 存储结果
    output_ptrs = output_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :]
    tl.store(output_ptrs, output, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.eps = eps
        self.negative_slope = negative_slope
        self.num_groups = num_groups
        self.hidden_size = hidden_size
    
    def forward(self, x):
        batch_size, input_size = x.shape
        hidden_size = self.hidden_size
        
        # 矩阵乘法: x @ W^T + bias
        fc_output = torch.empty((batch_size, hidden_size), dtype=x.dtype, device=x.device)
        
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        grid_mm = (
            triton.cdiv(batch_size, BLOCK_SIZE_M),
            triton.cdiv(hidden_size, BLOCK_SIZE_N),
        )
        
        matmul_kernel[grid_mm](
            x, self.fc.weight, fc_output, self.fc.bias,
            batch_size, hidden_size, input_size,
            x.stride(0), x.stride(1),
            self.fc.weight.stride(1), self.fc.weight.stride(0),
            fc_output.stride(0), fc_output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        # GroupNorm + LeakyReLU + Sum
        output = torch.empty((batch_size, hidden_size), dtype=x.dtype, device=x.device)
        
        BLOCK_SIZE_ROW = 32
        CHANNELS_PER_GROUP = hidden_size // self.num_groups  # 16
        
        grid_gn = (
            triton.cdiv(batch_size, BLOCK_SIZE_ROW),
            self.num_groups,
        )
        
        groupnorm_leakyrelu_sum_kernel[grid_gn](
            fc_output, output, self.gn.weight, self.gn.bias,
            batch_size, hidden_size, self.num_groups,
            fc_output.stride(0),
            self.eps,
            self.negative_slope,
            BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
            CHANNELS_PER_GROUP=CHANNELS_PER_GROUP,
        )
        
        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

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

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level2/62_Matmul_GroupNorm_LeakyReLU_Sum_weights.pt"
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
