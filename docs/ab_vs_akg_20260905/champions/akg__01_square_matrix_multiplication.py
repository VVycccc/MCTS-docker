import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def akg_01_square_matrix_multiplication_it5_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 主循环：沿 K 轴迭代
    for k in range(0, K, BLOCK_SIZE_K):
        # 创建块指针
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, k), 
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(k, pid_n * BLOCK_SIZE_N), 
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
        )
        
        # 加载数据块
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        # 矩阵乘加
        accumulator = tl.dot(a, b, acc=accumulator)
    
    # 存储结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"矩阵维度不匹配: {K} != {K2}"
        c = torch.empty((M, N), device=A.device, dtype=A.dtype)
        def grid(meta):
            return (
                triton.cdiv(M, meta['BLOCK_SIZE_M']),
                triton.cdiv(N, meta['BLOCK_SIZE_N']),
            )
        akg_01_square_matrix_multiplication_it5_kernel[grid](
            A, B, c,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            c.stride(0), c.stride(1),
        )
        return c


def get_init_inputs():
    return []


def get_inputs():
    M, K, N = 512, 512, 512
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    return [A, B]

# --- DirecTune shim (class.forward -> run adapter) ---
def run(A, B):
    _m = Model()
    return _m.forward(A, B)
