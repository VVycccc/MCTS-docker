import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_hw_kernel(
    x_ptr, out_ptr,
    N, C_in, H, W,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    base = x_ptr + pid_n * C_in * H * W + pid_c * H * W
    acc = 0.0
    
    for start in range(0, H * W, BLOCK_HW):
        off = start + tl.arange(0, BLOCK_HW)
        mask = off < H * W
        data = tl.load(base + off, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(data)
        
    tl.store(out_ptr + pid_n * C_in + pid_c, acc)


@triton.jit
def post_kernel(
    x_sum_ptr, weight_sum_ptr, conv_bias_ptr, user_bias_ptr, out_ptr,
    N, C_in, C_out, H_out, W_out,
    BLOCK_CIN: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    
    cin_off = tl.arange(0, BLOCK_CIN)
    cin_mask = cin_off < C_in
    x_sum = tl.load(x_sum_ptr + pid_n * C_in + cin_off, mask=cin_mask, other=0.0).to(tl.float32)
    
    cout_off = tl.arange(0, BLOCK_COUT)
    cout_mask = cout_off < C_out
    
    w_ptrs = weight_sum_ptr + cin_off[:, None] * C_out + cout_off[None, :]
    w_mask = cin_mask[:, None] & cout_mask[None, :]
    w_block = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
    
    matmul_res = tl.sum(x_sum[:, None] * w_block, axis=0)
    
    area = H_out * W_out
    conv_bias = tl.load(conv_bias_ptr + cout_off, mask=cout_mask, other=0.0).to(tl.float32)
    gap_out = (matmul_res + conv_bias * area) / area
    
    user_bias = tl.load(user_bias_ptr + cout_off, mask=cout_mask, other=0.0).to(tl.float32)
    bias_add = gap_out + user_bias
    
    max_val = tl.max(bias_add, axis=0)
    stable = bias_add - max_val
    exp_val = tl.exp(stable)
    sum_exp = tl.sum(exp_val, axis=0)
    lse = max_val + tl.log(sum_exp)
    
    result = lse * 10.0
    tl.store(out_ptr + pid_n, result)


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        torch.manual_seed(0)
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.clone())
        self.conv_bias = nn.Parameter(conv.bias.clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
    def forward(self, x):
        N, C_in, H, W = x.shape
        C_out = self.weight.shape[1]
        K = self.weight.shape[2]
        H_out = H + K - 1
        W_out = W + K - 1
        
        # 预计算权重和，形状为 (C_in, C_out)
        weight_sum = self.weight.sum(dim=(2, 3)).contiguous()
        
        # 分配中间结果和输出
        x_sum = torch.empty((N, C_in), dtype=torch.float32, device=x.device)
        output = torch.empty((N, 1), dtype=x.dtype, device=x.device)
        
        # 启动 Kernel 1: 计算输入在 H, W 维度上的和
        BLOCK_HW = 1024
        sum_hw_kernel[(N, C_in)](
            x, x_sum,
            N, C_in, H, W,
            BLOCK_HW=BLOCK_HW,
        )
        
        # 启动 Kernel 2: 融合的后续计算 (矩阵乘法 -> GAP -> BiasAdd -> LogSumExp -> Multiply)
        BLOCK_CIN = triton.next_power_of_2(C_in)
        BLOCK_COUT = triton.next_power_of_2(C_out)
        post_kernel[(N,)](
            x_sum, weight_sum, self.conv_bias, self.bias, output,
            N, C_in, C_out, H_out, W_out,
            BLOCK_CIN=BLOCK_CIN,
            BLOCK_COUT=BLOCK_COUT,
        )
        
        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        x = x + self.bias
        x = torch.logsumexp(x, dim=1, keepdim=True)  # Log-sum-exp
        x = torch.sum(x, dim=(2, 3))  # Sum
        x = x * 10.0  # Multiplication
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply_weights.pt"
_MODEL = None
def run(x, *args):
    global _MODEL
    if _MODEL is None:
        torch.manual_seed(0)
        _MODEL = ModelNew(*get_init_inputs())
        _ref = Model(*get_init_inputs())
        _ref.load_state_dict(torch.load(_weights_path, map_location='cpu', weights_only=True))
        _rp = list(_ref.parameters()); _np = list(_MODEL.parameters())
        for _pn, _pr in zip(_np, _rp):
            if _pn.shape == _pr.shape:
                _pn.data.copy_(_pr.data)
        _MODEL = _MODEL.to(x.device).eval()
    return _MODEL(x, *args)
