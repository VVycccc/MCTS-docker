import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        cumsum = torch.cumsum(x.narrow(dim=self.dim, start=0, length=x.size(self.dim)-1), dim=self.dim)
        return torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), cumsum), dim=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]


# --- EXPANDED REFERENCE ---
# run() 调用原始 Model.forward()（PyTorch 算，正确）。
# 写 Triton kernel 时：不要照抄 nn.Module，用 torch 实现（有权重则从 _weights_path 加载并缓存）。
import torch as _torch
_weights_path = "problems/kb_level1/92_cumsum_exclusive_weights.pt"
_model_cache = None
_model_device = None

def run(x):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(x.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache = _model_cache.to(x.device).eval()
        _model_device = str(x.device)
    return _model_cache(x)
