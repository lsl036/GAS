import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flash-topk-attention"))
sys.path.insert(0, str(ROOT))

from ftka.cuda_ops.gemv import sparse_batched_gemv_int4_k
from twilight.pyimpl.quantize import pack_min_max_int4_k_for_gemv


def benchmark(fn, warmups=20, repeats=100):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e6


def main():
    torch.manual_seed(0)
    device = "cuda"
    batch_size = 1
    num_heads = 32
    seq_len = 16384
    head_dim = 128

    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )
    packed, scales, zeros = pack_min_max_int4_k_for_gemv(k)

    for max_selected in [128, 256, 512, 1024, 2048, 4096, 8192]:
        selected_indices = torch.randint(
            0,
            seq_len,
            (batch_size, num_heads, max_selected),
            dtype=torch.int32,
            device=device,
        )
        selected_valid = torch.ones(
            batch_size, num_heads, max_selected, dtype=torch.uint8, device=device
        )
        out = torch.empty(
            batch_size, num_heads, max_selected, dtype=torch.float16, device=device
        )

        elapsed_us = benchmark(
            lambda: sparse_batched_gemv_int4_k(
                q, out, packed, scales, zeros, selected_indices, selected_valid
            )
        )
        print(f"max_selected={max_selected}, sparse_int4_gemv={elapsed_us:.3f} us")


if __name__ == "__main__":
    main()
