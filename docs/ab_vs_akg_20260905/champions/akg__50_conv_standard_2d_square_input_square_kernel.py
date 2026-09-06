import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def akg_50_conv_standard_2d_square_input_square_kernel_it5_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, K_h, K_w, stride, padding,
    H_out, W_out, K_total,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    oc_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    oc_mask = oc_offs < C_out

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_total = N * H_out * W_out
    n_mask = n_offs < n_total

    batch_idx = n_offs // (H_out * W_out)
    hw_idx = n_offs % (H_out * W_out)
    oh_idx = hw_idx // W_out
    ow_idx = hw_idx % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_total, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_total

        ic_idx = k_offs // (K_h * K_w)
        khw_idx = k_offs % (K_h * K_w)
        kh_idx = khw_idx // K_w
        kw_idx = khw_idx % K_w

        ih_idx = kh_idx[:, None] + oh_idx[None, :] * stride - padding
        iw_idx = kw_idx[:, None] + ow_idx[None, :] * stride - padding

        valid = (ih_idx >= 0) & (ih_idx < H) & (iw_idx >= 0) & (iw_idx < W)
        valid = valid & k_mask[:, None] & n_mask[None, :]

        in_off = (batch_idx[None, :] * (C_in * H * W) +
                  ic_idx[:, None] * (H * W) +
                  ih_idx * W + iw_idx)
        x = tl.load(x_ptr + in_off, mask=valid, other=0.0)

        w_off = (oc_offs[:, None] * K_total +
                 ic_idx[None, :] * (K_h * K_w) +
                 kh_idx[None, :] * K_w + kw_idx[None, :])
        wgt = tl.load(w_ptr + w_off, mask=oc_mask[:, None] & k_mask[None, :], other=0.0)

        acc += tl.dot(wgt, x)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    out_off = (batch_idx[None, :] * (C_out * H_out * W_out) +
               oc_offs[:, None] * (H_out * W_out) +
               oh_idx[None, :] * W_out + ow_idx[None, :])
    tl.store(out_ptr + out_off, acc, mask=oc_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        torch.manual_seed(0)
        conv = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone())
        self.stride = 4
        self.padding = 2
        self.K_h = 11
        self.K_w = 11
        self.C_in = 3
        self.C_out = 96

    def forward(self, x):
        N, C_in, H, W = x.shape
        C_out = self.C_out
        K_h, K_w = self.K_h, self.K_w
        stride = self.stride
        padding = self.padding

        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1
        K_total = C_in * K_h * K_w

        output = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (triton.cdiv(C_out, BLOCK_M), triton.cdiv(N * H_out * W_out, BLOCK_N))

        akg_50_conv_standard_2d_square_input_square_kernel_it5_kernel[grid](
            x, self.weight, self.bias, output,
            N, C_in, H, W, C_out, K_h, K_w, stride, padding,
            H_out, W_out, K_total,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        return output

# --- DirecTune shim (.pt weights aligned, cached) ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        x = self.conv1(x)
        return x

# Test code
batch_size = 256
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]

_weights_path = "/home/wangyichen/DirecTune-MCTS/problems/kb_level1/50_conv_standard_2d_square_input_square_kernel_weights.pt"
_MODEL = None
def run(x, *args):
    global _MODEL
    if _MODEL is None:
        import torch
        torch.manual_seed(0)
        _MODEL = ModelNew(*get_init_inputs())
        _ref = Model(*get_init_inputs())
        _ref.load_state_dict(torch.load(_weights_path, map_location='cpu', weights_only=True))
        _rp = list(_ref.parameters()); _np = list(_MODEL.parameters())
        for _pn, _pr in zip(_np, _rp):
            if _pn.shape == _pr.shape:
                _pn.data.copy_(_pr.data)
        _MODEL = _MODEL.to(x.device).eval()
    return _MODEL(x, *args)
