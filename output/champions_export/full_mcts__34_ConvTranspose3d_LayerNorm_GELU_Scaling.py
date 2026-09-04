import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def ln_gelu_scale_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    n_rows,
    eps, scaling_factor,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """
    融合了 LayerNorm + GELU + Scaling 的 Triton Kernel
    每个程序处理 ROWS_PER_PROGRAM 行数据
    """
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM
    cols = tl.arange(0, BLOCK_SIZE)
    
    # 预加载 LayerNorm 的权重和偏置（所有行共享，只需加载一次）
    w = tl.load(w_ptr + cols).to(tl.float32)
    b = tl.load(b_ptr + cols).to(tl.float32)
    
    sqrt_2 = 1.4142135623730951
    
    for i in range(ROWS_PER_PROGRAM):
        row = row_start + i
        if row < n_rows:
            x = tl.load(x_ptr + row * N_COLS + cols).to(tl.float32)
            
            # LayerNorm: 计算均值和方差
            mean = tl.sum(x, axis=0) / N_COLS
            x_centered = x - mean
            var = tl.sum(x_centered * x_centered, axis=0) / N_COLS
            rstd = 1.0 / tl.sqrt(var + eps)
            
            # 归一化 + 仿射变换 + GELU + 缩放
            ln_out = x_centered * rstd * w + b
            gelu_out = 0.5 * ln_out * (1.0 + tl.erf(ln_out / sqrt_2))
            tl.store(out_ptr + row * N_COLS + cols, gelu_out * scaling_factor)


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        # 固定随机种子，确保与原始 Model 权重一致
        torch.manual_seed(0)
        
        # 创建 3D 转置卷积层
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
        # 创建 LayerNorm 层以获取其权重和偏置
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        
        self.scaling_factor = scaling_factor
        self.eps = eps

        # 启用 TF32 加速卷积和 cuDNN auto-tune
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        # 1. 执行 3D 转置卷积
        x = self.conv_transpose(x)
        
        # 2. 准备融合 LayerNorm + GELU + Scaling
        # 形状变换为 2D 以便处理
        B, C, D, H, W = x.shape
        x = x.reshape(-1, C)
        n_rows, n_cols = x.shape
        
        # 分配输出张量
        out = torch.empty_like(x)
        
        # 计算块大小 (大于等于 n_cols 的最小 2 的幂)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        # 启动 Triton Kernel (每个 program 处理多行以减少 launch 开销)
        ROWS_PER_PROGRAM = 4
        grid = (triton.cdiv(n_rows, ROWS_PER_PROGRAM),)
        ln_gelu_scale_kernel[grid](
            x, self.layer_norm.weight, self.layer_norm.bias, out,
            n_rows,
            self.eps, self.scaling_factor,
            N_COLS=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
            num_warps=1,
        )
        
        # 恢复原始形状
        out = out.reshape(B, C, D, H, W)
        return out

# --- DirecTune shim (.pt weights aligned, cached) ---

batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/34_ConvTranspose3d_LayerNorm_GELU_Scaling_weights.pt"
_MODEL = None
def run(x, *args):
    global _MODEL
    if _MODEL is None:
        torch.manual_seed(0)
        _MODEL = ModelNew(*get_init_inputs())
        _MODEL.load_state_dict(torch.load(_weights_path, map_location='cpu', weights_only=True))
        _MODEL = _MODEL.to(x.device).eval()
    return _MODEL(x, *args)
