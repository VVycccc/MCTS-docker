import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softsign activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softsign activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        return x / (1 + torch.abs(x))

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed

# --- EXPANDED REFERENCE ---
# run() 调用原始 Model.forward()（PyTorch 算，正确）。
# 写 Triton kernel 时：不要照抄 nn.Module，用 torch 实现（有权重则从 _weights_path 加载并缓存）。
import torch as _torch
_weights_path = "problems/kb_level1/30_softsign_weights.pt"
_model_cache = None
_model_device = None

def run(x):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(x.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache = _model_cache.to(x.device).eval()
        _model_device = str(x.device)
    return _model_cache(x)
