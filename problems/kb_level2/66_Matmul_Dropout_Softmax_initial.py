import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.dropout(x)
        x = torch.softmax(x, dim=1)  # Softmax over features
        return x

batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, dropout_p]

# Frozen weights (loaded from .pt at module init):
#   matmul.weight: [16384, 16384] (float32)
#   matmul.bias: [16384] (float32)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/66_Matmul_Dropout_Softmax_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    matmul_bias = _weights['matmul.bias']  # [16384]
    matmul_weight = _weights['matmul.weight']  # [16384, 16384]

    x = _torch.nn.functional.linear(x, matmul_weight, matmul_bias)
    x = _torch.softmax(x, dim=-1)
    return x

# --- UNIVERSAL RUN (验证用，调 Model.forward) ---
_model_cache = None
_model_device = None

def run(x):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(x.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache.load_state_dict(_torch.load(_weights_path, map_location='cpu', weights_only=True))
        _model_cache = _model_cache.to(x.device).eval()
        _model_device = str(x.device)
    return _model_cache(x)
