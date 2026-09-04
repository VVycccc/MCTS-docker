import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def qk_kernel(
    Q, K, S,
    M, N, D,
    stride_q_bh, stride_q_m, stride_q_d,
    stride_k_bh, stride_k_n, stride_k_d,
    stride_s_bh, stride_s_m, stride_s_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    bh = pid // (num_pid_m * num_pid_n)
    pid_mn = pid % (num_pid_m * num_pid_n)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k in range(0, tl.cdiv(D, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        q = tl.load(
            Q + bh * stride_q_bh + offs_m[:, None] * stride_q_m + offs_k[None, :] * stride_q_d,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < D),
            other=0.0,
        )
        k_tile = tl.load(
            K + bh * stride_k_bh + offs_n[None, :] * stride_k_n + offs_k[:, None] * stride_k_d,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < D),
            other=0.0,
        )
        acc += tl.dot(q, k_tile)

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    acc = acc * scale

    s_ptrs = S + bh * stride_s_bh + offs_m[:, None] * stride_s_m + offs_n[None, :] * stride_s_n
    tl.store(s_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def softmax_kernel(
    S, P,
    M, N,
    stride_s_bh, stride_s_m, stride_s_n,
    stride_p_bh, stride_p_m, stride_p_n,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // M
    m = pid % M

    offs_n = tl.arange(0, BLOCK_N)

    row_max = float('-inf')
    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs = n_start * BLOCK_N + offs_n
        val = tl.load(
            S + bh * stride_s_bh + m * stride_s_m + offs * stride_s_n,
            mask=offs < N,
            other=float('-inf'),
        )
        row_max = tl.maximum(row_max, tl.max(val, axis=0))

    row_sum = 0.0
    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs = n_start * BLOCK_N + offs_n
        val = tl.load(
            S + bh * stride_s_bh + m * stride_s_m + offs * stride_s_n,
            mask=offs < N,
            other=float('-inf'),
        )
        row_sum += tl.sum(tl.exp(val - row_max), axis=0)

    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs = n_start * BLOCK_N + offs_n
        val = tl.load(
            S + bh * stride_s_bh + m * stride_s_m + offs * stride_s_n,
            mask=offs < N,
            other=float('-inf'),
        )
        out = tl.exp(val - row_max) / row_sum
        tl.store(
            P + bh * stride_p_bh + m * stride_p_m + offs * stride_p_n,
            out,
            mask=offs < N,
        )


@triton.jit
def pv_kernel(
    P, V, O,
    M, N, D,
    stride_p_bh, stride_p_m, stride_p_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_o_bh, stride_o_m, stride_o_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_d = tl.cdiv(D, BLOCK_K)
    bh = pid // (num_pid_m * num_pid_d)
    pid_md = pid % (num_pid_m * num_pid_d)
    pid_m = pid_md // num_pid_d
    pid_d = pid_md % num_pid_d

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_K], tl.float32)
    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n_start * BLOCK_N + tl.arange(0, BLOCK_N)
        p = tl.load(
            P + bh * stride_p_bh + offs_m[:, None] * stride_p_m + offs_n[None, :] * stride_p_n,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        )
        v = tl.load(
            V + bh * stride_v_bh + offs_n[:, None] * stride_v_n + offs_d[None, :] * stride_v_d,
            mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0,
        )
        acc += tl.dot(p, v)

    o_ptrs = O + bh * stride_o_bh + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))


def run(Q, K, V):
    B, H, M, D = Q.shape
    N = K.shape[2]
    BH = B * H

    Q_ = Q.reshape(BH, M, D).contiguous()
    K_ = K.reshape(BH, N, D).contiguous()
    V_ = V.reshape(BH, N, D).contiguous()

    S = torch.empty(BH, M, N, device=Q.device, dtype=torch.float32)
    P = torch.empty(BH, M, N, device=Q.device, dtype=torch.float32)
    O = torch.empty(BH, M, D, device=Q.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid1 = (BH * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    qk_kernel[grid1](
        Q_, K_, S,
        M, N, D,
        Q_.stride(0), Q_.stride(1), Q_.stride(2),
        K_.stride(0), K_.stride(1), K_.stride(2),
        S.stride(0), S.stride(1), S.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    grid2 = (BH * M,)
    softmax_kernel[grid2](
        S, P,
        M, N,
        S.stride(0), S.stride(1), S.stride(2),
        P.stride(0), P.stride(1), P.stride(2),
        BLOCK_N=BLOCK_N,
    )

    grid3 = (BH * triton.cdiv(M, BLOCK_M) * triton.cdiv(D, BLOCK_K),)
    pv_kernel[grid3](
        P, V_, O,
        M, N, D,
        P.stride(0), P.stride(1), P.stride(2),
        V_.stride(0), V_.stride(1), V_.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return O.reshape(B, H, M, D)
