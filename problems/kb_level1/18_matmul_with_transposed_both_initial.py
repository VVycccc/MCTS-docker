import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A.T, B.T)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(K, M)
    B = torch.rand(N, K)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed

# --- EXPANDED REFERENCE ---
# run() 调用原始 Model.forward()（PyTorch 算，正确）。
# 写 Triton kernel 时：不要照抄 nn.Module，用 torch 实现（有权重则从 _weights_path 加载并缓存）。
import torch as _torch
_weights_path = "problems/kb_level1/18_matmul_with_transposed_both_weights.pt"
_model_cache = None
_model_device = None

def run(A, B):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(A.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache = _model_cache.to(A.device).eval()
        _model_device = str(A.device)
    return _model_cache(A, B)
