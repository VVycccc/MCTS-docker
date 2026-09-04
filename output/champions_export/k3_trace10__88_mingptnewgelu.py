import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 每线程 4 元素(128-bit 向量化)，grid 最大、SM 最易饱和、负载均衡最好
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        # 每线程 4 元素(128-bit 向量化)，当前基线组合
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        # 每线程 16 元素(4 条独立 float4 load，访存 ILP 高)，grid 最小、调度开销低
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr, EVEN_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 1D 平坦视图 + 连续 offs：合并访问；BLOCK_SIZE/(num_warps*32)==4 时
    # 每线程处理 4 个连续 float，编译器可生成 128-bit 向量化 load/store
    if EVEN_N:
        x = tl.load(x_ptr + offs)
    else:
        x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    # GELU (tanh approximation):
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # 常数折叠: A = sqrt(2/pi), B = sqrt(2/pi)*0.044715，inner 用 Horner 形式
    # x*(A + B*x*x)，比 x + c*x*x*x 少一次逐元素乘法（编译器可融合为 FMA）
    inner = x * (0.7978845608028654 + 0.0356774081363001 * (x * x))
    # 0.5*(1+tanh(u)) == sigmoid(2u)，sigmoid(v) = 1/(1+exp2(-v*log2(e)))
    # 折叠 K = 2*log2(e) = 2.8853900817779268，单条 ex2.approx 快速路径，
    # 替代 sigmoid 内部的 mul+exp+div 及后续 2*s-1、0.5*x*(1+t) 一串指令；
    # inner 极大负时 exp2 上溢为 inf -> y=0，极大正时 exp2->0 -> y=x，语义安全
    y = x / (1.0 + tl.exp2(-2.8853900817779268 * inner))
    if EVEN_N:
        tl.store(y_ptr + offs, y)
    else:
        tl.store(y_ptr + offs, y, mask=offs < N)


def run(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    N = x.numel()
    # x 已 contiguous，按 1D 平坦视图处理；BLOCK_SIZE/num_warps 由 autotune 实测选取。
    # EVEN_N 必须对所有候选 BLOCK_SIZE(512/1024/4096) 同时成立：取其公倍数 4096 判定
    # (N=2^26 可被整除)，保证无尾块的 EVEN_N 分支在任意 autotune 配置下都正确。
    EVEN_N = (N % 4096) == 0
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, N, EVEN_N=EVEN_N)
    return y
