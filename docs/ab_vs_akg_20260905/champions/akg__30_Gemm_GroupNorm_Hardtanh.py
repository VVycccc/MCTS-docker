import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
    restore_value=['c_ptr'],
)
@triton.jit
def gemm_kernel(
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
    GEMM kernel: 计算 x @ weight.T + bias
    x: (M, K), weight: (N, K) -> weight.T: (K, N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 沿K轴迭代计算矩阵乘法
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
    bias = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0).to(tl.float32)
    accumulator += bias[None, :]
    
    # 存储GEMM结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


@triton.jit
def group_norm_hardtanh_kernel(
    input_ptr, output_ptr, gamma_ptr, beta_ptr,
    n_rows, n_cols, num_groups,
    stride_in, stride_out,
    eps,
    min_val, max_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GroupNorm + Hardtanh kernel
    每个程序处理一行的一个group
    """
    pid_row = tl.program_id(0)
    pid_group = tl.program_id(1)
    
    # 计算当前group的范围
    group_size = n_cols // num_groups
    col_start = pid_group * group_size
    col_end = col_start + group_size
    
    # 创建列偏移和掩码
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < col_end
    
    # 加载数据
    row_ptr = input_ptr + pid_row * stride_in + cols
    data = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # 计算均值
    mean = tl.sum(data, axis=0) / group_size
    
    # 计算方差
    diff = tl.where(mask, data - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / group_size
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 归一化
    normalized = (data - mean) * rstd
    
    # 应用gamma和beta（仿射变换）
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    output = normalized * gamma + beta
    
    # Hardtanh: 将值限制在[min_val, max_val]范围内
    output = tl.where(output < min_val, min_val, output)
    output = tl.where(output > max_val, max_val, output)
    
    # 存储结果
    out_row_ptr = output_ptr + pid_row * stride_out + cols
    tl.store(out_row_ptr, output, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        # 固定随机种子，确保权重与原始Model一致
        torch.manual_seed(0)
        
        # 创建Linear层并提取权重
        linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(linear.weight.clone())
        self.bias = nn.Parameter(linear.bias.clone())
        
        # 创建GroupNorm并提取gamma和beta
        group_norm = nn.GroupNorm(num_groups, out_features)
        self.gamma = nn.Parameter(group_norm.weight.clone())
        self.beta = nn.Parameter(group_norm.bias.clone())
        
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.eps = 1e-5
    
    def forward(self, x):
        """
        执行 GEMM -> GroupNorm -> Hardtanh
        x: (batch_size, in_features)
        """
        M, K = x.shape
        N = self.weight.shape[0]
        
        # 第一步：GEMM (x @ weight.T + bias)
        gemm_output = torch.empty((M, N), dtype=torch.float32, device=x.device)
        
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N']),
        )
        gemm_kernel[grid](
            x, self.weight, gemm_output, self.bias,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight.stride(1), self.weight.stride(0),  # weight转置
            gemm_output.stride(0), gemm_output.stride(1),
        )
        
        # 第二步：GroupNorm + Hardtanh
        output = torch.empty((M, N), dtype=x.dtype, device=x.device)
        group_size = N // self.num_groups
        BLOCK_SIZE = triton.next_power_of_2(group_size)
        
        grid2 = (M, self.num_groups)
        group_norm_hardtanh_kernel[grid2](
            gemm_output, output, self.gamma, self.beta,
            M, N, self.num_groups,
            gemm_output.stride(0), output.stride(0),
            self.eps,
            self.hardtanh_min, self.hardtanh_max,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.group_norm(x)
        x = self.hardtanh(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level2/30_Gemm_GroupNorm_Hardtanh_weights.pt"
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
