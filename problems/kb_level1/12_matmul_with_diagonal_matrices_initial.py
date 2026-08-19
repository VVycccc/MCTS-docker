import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        # Logically equivalent to torch.diag(A) @ B 
        # more efficient as no need to materialize a full N×N matrix
        return A.unsqueeze(1) * B

M = 4096
N = 4096

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed

# --- EXPANDED REFERENCE ---
# run() 调用原始 Model.forward()（PyTorch 算，正确）。
# 写 Triton kernel 时：不要照抄 nn.Module，用 torch 实现（有权重则从 _weights_path 加载并缓存）。
import torch as _torch
_weights_path = "problems/kb_level1/12_matmul_with_diagonal_matrices_weights.pt"
_model_cache = None
_model_device = None

def run(A, B):
    global _model_cache, _model_device
    if _model_cache is None or _model_device != str(A.device):
        _model_cache = Model(*get_init_inputs())
        _model_cache = _model_cache.to(A.device).eval()
        _model_device = str(A.device)
    return _model_cache(A, B)
