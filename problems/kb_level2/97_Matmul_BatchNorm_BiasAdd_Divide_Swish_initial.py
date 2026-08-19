import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        x = x + self.bias
        x = x / self.divide_value
        x = x * torch.sigmoid(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]
# Frozen weights (loaded from .pt at module init):
#   bias: [1] (float32)
#   matmul.weight: [8192, 8192] (float32)
#   matmul.bias: [8192] (float32)
#   bn.weight: [8192] (float32)
#   bn.bias: [8192] (float32)
#   bn.running_mean: [8192] (float32)
#   bn.running_var: [8192] (float32)
#   bn.num_batches_tracked: [] (int64)

# --- EXPANDED REFERENCE (函数式，给 LLM 看) ---
# Model.forward() 的函数式展开（_weights + _torch.nn.functional.*）。
# LLM 写 Triton 时参考此函数式写法（torch.load(_weights_path) + 缓存 + 函数式计算）。
# 验证用下方的 run()（通用，调 Model），不要照抄 nn.Module。

import torch as _torch
_weights_path = "/home/wangyichen/DirecTune/problems/kb_level2/97_Matmul_BatchNorm_BiasAdd_Divide_Swish_weights.pt"
_weights = None
_device = None

def _init_weights(device):
    global _weights, _device
    _weights = {k: v.to(device) for k, v in _torch.load(_weights_path, map_location='cpu', weights_only=True).items()}

def ref_expanded(x):
    if _weights is None or str(next(iter(_weights.values())).device) != str(x.device):
        _init_weights(x.device)
    bias = _weights['bias']  # [1]
    bn_bias = _weights['bn.bias']  # [8192]
    bn_num_batches_tracked = _weights['bn.num_batches_tracked']  # []
    bn_running_mean = _weights['bn.running_mean']  # [8192]
    bn_running_var = _weights['bn.running_var']  # [8192]
    bn_weight = _weights['bn.weight']  # [8192]
    matmul_bias = _weights['matmul.bias']  # [8192]
    matmul_weight = _weights['matmul.weight']  # [8192, 8192]

    x = _torch.nn.functional.linear(x, matmul_weight, matmul_bias)
    x = _torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, weight=bn_weight, bias=bn_bias, training=True)
    x = x + bias
    x = x / 1.0
    x = _torch.sigmoid(x)
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
