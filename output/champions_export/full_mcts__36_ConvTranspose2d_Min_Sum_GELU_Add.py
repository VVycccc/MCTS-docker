import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_min_sum_gelu_add_kernel(
    x_ptr,          # (B, C, H, W) conv_transpose输出
    bias_ptr,       # (1,) 加法偏置
    output_ptr,     # (B, 1, 1, W) 最终输出
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    融合内核：min(dim=1) -> sum(dim=2) -> GELU -> add(bias)
    每个程序处理一个batch和一段width
    """
    pid_b = tl.program_id(0)
    pid_w = tl.program_id(1)

    # 当前程序处理的width偏移
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offsets < W

    # 累加器：沿H维度求和
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # 沿H维度迭代
    for h in range(H):
        # 初始化沿C维度的最小值
        min_val = tl.full((BLOCK_W,), float('inf'), dtype=tl.float32)

        # 沿C维度分块迭代，计算min over C
        for c_start in range(0, C, BLOCK_C):
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offsets < C

            # 加载 x[b, c, h, w] -> (BLOCK_C, BLOCK_W)
            ptrs = (x_ptr + pid_b * stride_b 
                    + c_offsets[:, None] * stride_c 
                    + h * stride_h 
                    + w_offsets[None, :] * stride_w)
            mask = c_mask[:, None] & w_mask[None, :]

            x = tl.load(ptrs, mask=mask, other=float('inf')).to(tl.float32)

            # 沿C维度(axis=0)求最小值
            block_min = tl.min(x, axis=0)
            min_val = tl.minimum(min_val, block_min)

        # 累加沿H维度的和
        acc += min_val

    # GELU激活: x * 0.5 * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475  # 1/sqrt(2)
    acc_gelu = acc * 0.5 * (1.0 + tl.erf(acc * inv_sqrt2))

    # 加偏置
    bias = tl.load(bias_ptr)
    result = acc_gelu + bias

    # 存储输出: output[b, 0, 0, w] = output_ptr[b * W + w]
    out_ptrs = output_ptr + pid_b * W + w_offsets
    tl.store(out_ptrs, result, mask=w_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # 固定随机种子，确保权重与原始Model一致
        torch.manual_seed(0)
        conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.weight = nn.Parameter(conv_transpose.weight.clone())
        self.bias_conv = nn.Parameter(conv_transpose.bias.clone()) if conv_transpose.bias is not None else None
        self.bias_add = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # 第一步：ConvTranspose2d（使用PyTorch实现）
        x = torch.nn.functional.conv_transpose2d(
            x, self.weight, self.bias_conv,
            self.stride, self.padding, self.output_padding
        )

        B, C, H, W = x.shape

        # 分配输出张量: (B, 1, 1, W)
        output = torch.empty((B, 1, 1, W), dtype=x.dtype, device=x.device)

        # 配置块大小
        BLOCK_W = 64
        BLOCK_C = 128  # C=128，一个块覆盖所有通道

        # 网格: (B, cdiv(W, BLOCK_W))
        grid = (B, triton.cdiv(W, BLOCK_W))

        # 启动融合内核
        fused_min_sum_gelu_add_kernel[grid](
            x, self.bias_add, output,
            B, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            BLOCK_W=BLOCK_W,
            BLOCK_C=BLOCK_C,
        )

        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum operation along channel dimension
        x = torch.sum(x, dim=2, keepdim=True)  # Sum operation along height dimension
        x = torch.nn.functional.gelu(x)  # GELU activation
        x = x + self.bias
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/36_ConvTranspose2d_Min_Sum_GELU_Add_weights.pt"
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
