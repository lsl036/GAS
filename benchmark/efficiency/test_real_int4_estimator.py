import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flash-topk-attention"))
sys.path.insert(0, str(ROOT))

from ftka.cuda_ops.gemv import batched_sparse_gemv_int4_k
from gas_attention.pyimpl.quantize import (
    pack_min_max_int4_k_for_gemv,
    dequantize_packed_int4_k_for_gemv,
)


def main():
    torch.manual_seed(0)
    device = "cuda"
    batch_size = 2
    num_heads = 4
    seq_len = 64
    head_dim = 128

    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )

    packed, scales, zeros = pack_min_max_int4_k_for_gemv(k)
    dequant = dequantize_packed_int4_k_for_gemv(
        packed, scales, zeros, batch_size, seq_len, num_heads, head_dim
    )
    ref = torch.matmul(q.unsqueeze(2), dequant.transpose(2, 3)).squeeze(2)

    out = torch.empty(batch_size, num_heads, seq_len, device=device, dtype=torch.float16)
    kv_page_indices = torch.arange(
        batch_size * seq_len, dtype=torch.int32, device=device
    )
    kv_page_indptr = torch.arange(
        batch_size + 1, dtype=torch.int32, device=device
    ) * seq_len
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    batched_sparse_gemv_int4_k(
        q,
        out,
        packed,
        scales,
        zeros,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )
    torch.cuda.synchronize()

    finite_ratio = torch.isfinite(out).float().mean().item()
    print(f"finite_ratio={finite_ratio:.4f}")
    print(f"out_min={out.nan_to_num().min().item():.6f}, out_max={out.nan_to_num().max().item():.6f}")

    max_err = (out - ref).abs().max().item()
    mean_err = (out - ref).abs().mean().item()
    print(f"max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    estimated_weights = out.unsqueeze(2) / math.sqrt(head_dim)
    assert estimated_weights.shape == (batch_size, num_heads, 1, seq_len)
    print("real int4 estimator test passed")


if __name__ == "__main__":
    main()
