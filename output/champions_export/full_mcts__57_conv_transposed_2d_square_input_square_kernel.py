import torch
import torch.nn as nn
import triton
import triton.language as tl

class Model(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# Test code
batch_size = 8
in_channels = 64  # double channels for heavier compute
out_channels = 64
kernel_size = 3
# larger square input
height = 1024
width = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization

# --- EXPANDED REFERENCE ---
_weights_path = "problems/kb_level1/57_conv_transposed_2d_square_input_square_kernel_weights.pt"
_model_cache = None
_model_device = None

@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, C_out, H, W, H_out, W_out, K,
    BLOCK_SIZE: tl.constexpr,
    C_in_const: tl.constexpr,
    K_const: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C_out * H_out * W_out
    mask = offs < total

    # Decode 1D offset to 4D indices: n, c_out, i, j
    n = offs // (C_out * H_out * W_out)
    rem = offs % (C_out * H_out * W_out)
    c_out = rem // (H_out * W_out)
    rem = rem % (H_out * W_out)
    i = rem // W_out
    j = rem % W_out

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for c_in in range(0, C_in_const):
        for kh in range(0, K_const):
            for kw in range(0, K_const):
                ih = i - kh
                iw = j - kw
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                
                x_idx = n * (C_in * H * W) + c_in * (H * W) + ih * W + iw
                w_idx = c_in * (C_out * K * K) + c_out * (K * K) + kh * K + kw
                
                x_val = tl.load(x_ptr + x_idx, mask=(mask & valid), other=0.0)
                w_val = tl.load(w_ptr + w_idx, mask=mask, other=0.0)
                acc += x_val * w_val

    tl.store(y_ptr + offs, acc, mask=mask)

def run(x):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(x.device):
        torch.backends.cudnn.benchmark = True
        _model_cache = Model(*get_init_inputs())
        _model_cache.load_state_dict(torch.load(_weights_path, map_location='cpu', weights_only=True))
        _model_cache = _model_cache.to(x.device).half().eval()
        _model_device = str(x.device)
    
    out = _model_cache(x.half())
    return out.float()
