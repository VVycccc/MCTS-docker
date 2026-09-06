import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def akg_l2_18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp_it2_kernel(
    x_ptr,          # 输入 x, shape (batch_size, in_features)
    w_sum_ptr,      # weight 沿 dim=0 求和, shape (in_features,)
    b_sum_ptr,      # bias 求和, scalar
    output_ptr,     # 输出, shape (batch_size, 1)
    n_features,     # in_features
    x_row_stride,   # x 的行步长
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 kernel: Matmul + Sum + Max + AvgPool + LogSumExp + LogSumExp
    
    分析:
    1. linear(x) = x @ W^T + b, shape (batch_size, out_features)
    2. sum(dim=1) -> shape (batch_size, 1)
    3. 之后 max/mean/logsumexp/logsumexp 在 dim=1 大小为1时都是恒等操作
       - max([v]) = v
       - mean([v]) = v  
       - logsumexp([v]) = log(exp(v)) = v
    
    利用数学等价性优化:
    sum_j(x @ W^T + b)[row, j] = sum_j sum_k x[row,k]*W[j,k] + sum_j b[j]
                               = sum_k x[row,k] * (sum_j W[j,k]) + sum_j b[j]
                               = x[row] @ weight_sum + bias_sum
    
    将 O(batch * in * out) 的矩阵乘法降为 O(batch * in) 的向量点积
    """
    row = tl.program_id(0)
    
    # 累加器，用于分块计算点积
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # 分块加载并计算点积
    for off in range(0, n_features, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_features
        
        # 加载输入数据和预计算的权重和
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_sum_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        # 累加逐元素乘积
        acc += x * w
    
    # 块内归约求和，完成点积计算
    dot_result = tl.sum(acc, axis=0)
    
    # 加上 bias_sum
    b_sum = tl.load(b_sum_ptr).to(tl.float32)
    result = dot_result + b_sum
    
    # 后续操作 (max, mean, logsumexp, logsumexp) 在 (batch_size, 1) 上都是恒等操作
    # 直接存储结果
    tl.store(output_ptr + row, result)


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 固定随机种子，确保与原始 Model 权重一致
        torch.manual_seed(0)
        linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(linear.weight.clone())
        self.bias = nn.Parameter(linear.bias.clone())
    
    def forward(self, x):
        """
        Args:
            x: shape (batch_size, in_features)
        Returns:
            output: shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        in_features = x.shape[1]
        
        # 预计算 weight_sum 和 bias_sum
        # weight shape: (out_features, in_features)
        # weight_sum = sum_j W[j, :], shape: (in_features,)
        # bias_sum = sum_j b[j], scalar
        weight_sum = self.weight.sum(dim=0)
        bias_sum = self.bias.sum()
        
        # 分配输出张量
        output = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)
        
        # 启动 kernel
        BLOCK_SIZE = 1024
        grid = (batch_size,)
        
        akg_l2_18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp_it2_kernel[grid](
            x, weight_sum, bias_sum, output,
            in_features,
            x.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)  # (batch_size, out_features)
        x = torch.sum(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.max(x, dim=1, keepdim=True)[0] # (batch_size, 1)
        x = torch.mean(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level2/18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp_weights.pt"
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
