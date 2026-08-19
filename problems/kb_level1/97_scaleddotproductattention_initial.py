import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out

batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs():
    Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    return [Q, K, V]

def get_init_inputs():
    return []


# --- EXPANDED REFERENCE ---
# run() 调用原始 Model.forward()（PyTorch 算，正确）。
# 写 Triton kernel 时：不要照抄 nn.Module，用 torch 实现（有权重则从 _weights_path 加载并缓存）。
import torch as _torch
_weights_path = "problems/kb_level1/97_scaleddotproductattention_weights.pt"
_model_cache = None
_model_device = None

def run(Q, K, V):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(Q.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache = _model_cache.to(Q.device).eval()
        _model_device = str(Q.device)
    return _model_cache(Q, K, V)
