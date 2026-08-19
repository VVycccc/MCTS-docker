import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices. 
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return torch.tril(torch.matmul(A, B))

M = 4096

def get_inputs():
    A = torch.rand(M, M)
    B = torch.rand(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed

# --- EXPANDED REFERENCE ---
# run() 调用原始 Model.forward()（PyTorch 算，正确）。
# 写 Triton kernel 时：不要照抄 nn.Module，用 torch 实现（有权重则从 _weights_path 加载并缓存）。
import torch as _torch
_weights_path = "problems/kb_level1/15_matmul_for_lower_triangular_matrices_weights.pt"
_model_cache = None
_model_device = None

def run(A, B):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(A.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache = _model_cache.to(A.device).eval()
        _model_device = str(A.device)
    return _model_cache(A, B)
