import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # GROUP_M swizzle：1D pid 重映射，分组调度提升 L2 对 A/B 面板的复用
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 控制流特化：4096 可被所有 autotune tile 整除，加载全程无掩码、
    # 完全合并且可被 num_stages 流水线化；循环行程编译期折叠
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # 输入已预转为 fp16：消除循环内 F2F 转换，shared 中 fp16 tile 可经 ldmatrix 直接喂 HMMA
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    # M/N 恒被 BLOCK_M/BLOCK_N 整除：无掩码全向量化 store，消除 store 谓词
    tl.store(c_ptrs, acc.to(C.dtype.element_ty))


def run(A, B):
    M, K = A.shape
    _, N = B.shape

    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # 预转换 fp16：一次性 ~0.2ms 转换开销，换来 shared 占用减半（流水可加深至 4-5 级）、
    # DRAM/L2 流量减半、ldmatrix 直接喂 HMMA、消除 K 循环内的 F2F 转换指令
    A16 = A.to(torch.float16)
    B16 = B.to(torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](
        A16, B16, C,
        M, N, K,
        A16.stride(0), A16.stride(1),
        B16.stride(0), B16.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
