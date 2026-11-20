import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flash-topk-attention"))
sys.path.insert(0, str(ROOT))

from ftka.cuda_ops.gemv import sparse_batched_gemv_int4_k
from twilight.pyimpl.quantize import (
    dequantize_packed_int4_k_for_gemv,
    pack_min_max_int4_k_for_gemv,
)


def main():
    torch.manual_seed(0)
    device = "cuda"
    batch_size = 2
    num_heads = 4
    seq_len = 64
    head_dim = 128
    max_selected = 17

    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )
    packed, scales, zeros = pack_min_max_int4_k_for_gemv(k)
    dequant = dequantize_packed_int4_k_for_gemv(
        packed, scales, zeros, batch_size, seq_len, num_heads, head_dim
    )
    ref = torch.matmul(q.unsqueeze(2), dequant.transpose(2, 3)).squeeze(2)

    selected_indices = torch.empty(
        batch_size, num_heads, max_selected, dtype=torch.int32, device=device
    )
    selected_valid = torch.ones(
        batch_size, num_heads, max_selected, dtype=torch.uint8, device=device
    )
    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            perm = torch.randperm(seq_len, device=device)[:max_selected].sort().values
            selected_indices[batch_idx, head_idx] = perm.to(torch.int32)

    out = torch.empty(
        batch_size, num_heads, max_selected, dtype=torch.float16, device=device
    )
    sparse_batched_gemv_int4_k(
        q,
        out,
        packed,
        scales,
        zeros,
        selected_indices,
        selected_valid,
    )
    torch.cuda.synchronize()

    ref_selected = torch.gather(ref, dim=-1, index=selected_indices.long())
    max_err = (out - ref_selected).abs().max().item()
    mean_err = (out - ref_selected).abs().mean().item()
    print(f"max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    torch.testing.assert_close(out, ref_selected, rtol=5e-2, atol=5e-2)
    print("sparse real int4 estimator test passed")


if __name__ == "__main__":
    main()
