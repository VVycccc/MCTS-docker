import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return torch.argmax(x, dim=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]

# --- EXPANDED REFERENCE ---
# run() 调用原始 Model.forward()（PyTorch 算，正确）。
# 写 Triton kernel 时：不要照抄 nn.Module，用 torch 实现（有权重则从 _weights_path 加载并缓存）。
import torch as _torch
_weights_path = "problems/kb_level1/51_argmax_over_a_dimension_weights.pt"
_model_cache = None
_model_device = None

def run(x):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(x.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache = _model_cache.to(x.device).eval()
        _model_device = str(x.device)
    return _model_cache(x)
