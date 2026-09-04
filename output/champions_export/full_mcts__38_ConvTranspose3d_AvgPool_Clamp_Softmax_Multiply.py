import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def clamp_softmax_mul_kernel(
    x_ptr, output_ptr, scale_ptr,
    n_spatial,
    stride_b, stride_c,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 Clamp + Softmax + Multiply 内核
    每个程序处理一个 的所有 spatial 元素
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # 计算基础偏移
    base_offset = pid_b * stride_b + pid_c * stride_c
    
    # 第一遍：online softmax 同时计算 max 和 sum（合并原第一、二遍）
    _max = -float('inf')
    _sum = 0.0
    for off in range(0, n_spatial, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_spatial
        x = tl.load(x_ptr + base_offset + cols, mask=mask, other=-float('inf')).to(tl.float32)
        x = tl.maximum(tl.minimum(x, clamp_max), clamp_min)
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(_max, block_max)
        exp_x = tl.exp(x - new_max)
        block_sum = tl.sum(exp_x, axis=0)
        _sum = _sum * tl.exp(_max - new_max) + block_sum
        _max = new_max
    
    # 第二遍：计算 softmax 并乘以 scale
    scale = tl.load(scale_ptr + pid_c).to(tl.float32)
    for off in range(0, n_spatial, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_spatial
        x = tl.load(x_ptr + base_offset + cols, mask=mask, other=-float('inf')).to(tl.float32)
        x = tl.maximum(tl.minimum(x, clamp_max), clamp_min)
        exp_x = tl.exp(x - _max)
        softmax_x = exp_x / _sum
        result = softmax_x * scale
        tl.store(output_ptr + base_offset + cols, result, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        torch.manual_seed(0)  # 固定种子，确保与原始Model权重一致
        self.pool_kernel_size = pool_kernel_size
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
    
    def forward(self, x):
        # Manual average pooling (replacing nn pooling helper)
        k = self.pool_kernel_size
        B, C, D, H, W = x.shape
        x = x.view(B, C, D // k, k, H // k, k, W // k, k)
        x = x.mean(dim=(3, 5, 7))
        
        # ConvTranspose3d
        x = self.conv_transpose(x)
        # 确保连续
        x = x.contiguous()
        
        b, c, d, h, w = x.shape
        n_spatial = d * h * w
        
        # 分配输出
        output = torch.empty_like(x)
        
        # 启动 kernel
        grid = (b, c)
        clamp_softmax_mul_kernel[grid](
            x, output, self.scale,
            n_spatial,
            x.stride(0), x.stride(1),
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
            BLOCK_SIZE=4096,
            num_warps=8,
        )
        
        return output


batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply_weights.pt"

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]

_MODEL = None

def run(x, *args):
    global _MODEL
    if _MODEL is None or str(next(_MODEL.parameters()).device) != str(x.device):
        torch.manual_seed(0)
        _MODEL = ModelNew(*get_init_inputs())
        _w = torch.load(_weights_path, map_location='cpu', weights_only=True)
        _MODEL.conv_transpose.weight.data.copy_(_w['conv_transpose.weight'])
        _MODEL.conv_transpose.bias.data.copy_(_w['conv_transpose.bias'])
        _MODEL.scale.data.copy_(_w['scale'])
        _MODEL = _MODEL.to(x.device).eval()
    return _MODEL(x, *args)
