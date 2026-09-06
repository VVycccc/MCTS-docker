import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_pool2d_kernel(
    x_ptr, output_ptr,
    N, C, H, W,
    pooled_h, pooled_w,
    stride, padding, dilation,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Max Pooling 2D Triton Kernel
    每个程序处理一个 (b, c, ph) 行的所有 pw 输出
    """
    # 获取程序 ID 并解码为
    pid = tl.program_id(0)
    bc_id = pid // pooled_h
    ph = pid % pooled_h
    b = bc_id // C
    c = bc_id % C

    # 输出列偏移
    pw_offsets = tl.arange(0, BLOCK_SIZE)
    pw_mask = pw_offsets < pooled_w

    # 初始化结果为负无穷
    result = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)

    # 遍历 kernel 区域
    for kh in range(KERNEL_SIZE):
        ih = ph * stride - padding + kh * dilation
        for kw in range(KERNEL_SIZE):
            iw = pw_offsets * stride - padding + kw * dilation
            # 边界检查
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & pw_mask
            # 计算输入偏移
            x_offsets = b * C * H * W + c * H * W + ih * W + iw
            # 加载数据，无效位置填充负无穷
            x_vals = tl.load(x_ptr + x_offsets, mask=valid, other=-float('inf'))
            # 取最大值
            result = tl.maximum(result, x_vals)

    # 计算输出偏移并存储
    out_offsets = b * C * pooled_h * pooled_w + c * pooled_h * pooled_w + ph * pooled_w + pw_offsets
    tl.store(output_ptr + out_offsets, result, mask=pw_mask)


class ModelNew(torch.nn.Module):
    """
    使用 Triton 实现的 Max Pooling 2D
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入连续
        x = x.contiguous()
        N, C, H, W = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation

        # 计算输出尺寸
        pooled_h = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        pooled_w = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # 分配输出张量
        output = torch.empty((N, C, pooled_h, pooled_w), dtype=x.dtype, device=x.device)

        # 块大小为大于 pooled_w 的最小 2 的幂
        BLOCK_SIZE = triton.next_power_of_2(pooled_w)

        # 网格大小：每个程序处理一行输出
        grid = (N * C * pooled_h,)

        # 启动内核
        max_pool2d_kernel[grid](
            x, output,
            N, C, H, W,
            pooled_h, pooled_w,
            stride, padding, dilation,
            KERNEL_SIZE=kernel_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool(x)

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level1/42_max_pooling_2d_weights.pt"
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
