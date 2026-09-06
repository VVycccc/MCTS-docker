import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton Gemm + Add + ReLU 内核
# 计算 C = ReLU(A @ B + bias)
# 其中 A 是 (M, K), B 是 (K, N), bias 是 (N,)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
    restore_value=['c_ptr'],
)
@triton.jit
def akg_l2_76_Gemm_Add_ReLU_it16_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton Gemm + Add + ReLU 内核
    每个程序实例处理输出矩阵的一个 (BLOCK_M, BLOCK_N) 块
    """
    # 获取程序 ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 主循环：沿 K 轴迭代
    for k in range(0, K, BLOCK_K):
        # 创建块指针
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, k),
            block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(k, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N), order=(1, 0)
        )
        
        # 加载数据块
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        # 矩阵乘加
        accumulator = tl.dot(a, b, acc=accumulator)
    
    # 加上 bias
    bias_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias_mask = bias_offs < N
    bias = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0)
    accumulator += bias[None, :]
    
    # ReLU 激活
    accumulator = tl.maximum(accumulator, 0.0)
    
    # 存储结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


class ModelNew(torch.nn.Module):
    """
    使用 Triton 实现的 Gemm + Add + ReLU 模型
    """
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        # 固定随机种子，确保与原始 Model 权重一致
        torch.manual_seed(0)
        # 创建 Linear 层并提取权重（bias=False，与原始 Model 一致）
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        # 单独创建 bias 参数（与原始 Model 一致）
        self.bias = nn.Parameter(torch.randn(bias_shape))
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        M, K = x.shape
        # nn.Linear 的 weight shape 是 (out_features, in_features) = (N, K)
        # kernel 中需要 B 的 shape 为 (K, N)，所以使用 weight.T
        w = self.gemm.weight.T  # (K, N)
        N = w.shape[1]
        
        # 分配输出张量
        c = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        # 定义网格
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        
        # 启动内核
        akg_l2_76_Gemm_Add_ReLU_it16_kernel[grid](
            x, w, c, self.bias,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            c.stride(0), c.stride(1),
        )
        
        return c

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x + self.bias
        x = torch.relu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape]

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level2/76_Gemm_Add_ReLU_weights.pt"
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
