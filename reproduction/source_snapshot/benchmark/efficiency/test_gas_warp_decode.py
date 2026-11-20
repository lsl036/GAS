"""GPU correctness tests for experimental GAS attention kernels."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from bench_e6_qasper_trace import reference

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def module():
    return load(name="gas_warp_decode_v1", sources=[str(ROOT / "csrc/src/gas_warp_decode.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


@pytest.mark.parametrize("mode,tile", [(0, 16), (1, 16), (2, 8), (2, 16), (2, 32), (3, 8), (3, 16), (4, 16), (5, 16), (6, 16), (7, 16), (8, 16), (9, 16), (10, 16), (11, 16)])
@pytest.mark.parametrize("case", ["mixed", "no_shared", "no_exceptions"])
def test_reference(module, mode, tile, case):
    torch.manual_seed(42)
    batch, heads, kv_heads, length = 2, 32, 8, 37
    q = torch.randn(batch, heads, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, kv_heads, length, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    sw = 0 if case == "no_shared" else 19
    ew = 0 if case == "no_exceptions" else 7
    shared = torch.full((batch, kv_heads, sw), -1, device="cuda", dtype=torch.int32)
    exceptions = torch.full((batch, heads, ew), -1, device="cuda", dtype=torch.int32)
    mask = torch.zeros((batch, heads, length), device="cuda", dtype=torch.bool)
    for b in range(batch):
        for g in range(kv_heads):
            selected = list(range(g % 4, g % 4 + (1 + (b + g) % 12))) if sw else []
            if selected:
                shared[b, g, :len(selected)] = torch.tensor(selected, device="cuda", dtype=torch.int32)
            for h in range(4 * g, 4 * g + 4):
                extra = list(range(25 + h % 4, 25 + h % 4 + (1 + (b + h) % 3))) if ew else []
                if extra:
                    exceptions[b, h, :len(extra)] = torch.tensor(extra, device="cuda", dtype=torch.int32)
                mask[b, h, selected + extra] = True
    output = module.decode(q, k, v, shared, exceptions, mode, tile)
    torch.cuda.synchronize()
    expected = reference(q, k, v, mask)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected, atol=0.003, rtol=0.003)


def test_invalid_mode_rejected(module):
    q = torch.zeros((1, 4, 128), device="cuda", dtype=torch.float16)
    k = torch.zeros((1, 1, 2, 128), device="cuda", dtype=torch.float16)
    shared = torch.zeros((1, 1, 1), device="cuda", dtype=torch.int32)
    ex = torch.empty((1, 4, 0), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="invalid mode"):
        module.decode(q, k, k, shared, ex, 12, 8)


def test_vector_unaligned_input_rejected(module):
    q = torch.zeros(4 * 128 + 1, device="cuda", dtype=torch.float16)[1:].view(1, 4, 128)
    k = torch.zeros((1, 1, 2, 128), device="cuda", dtype=torch.float16)
    shared = torch.zeros((1, 1, 1), device="cuda", dtype=torch.int32)
    ex = torch.empty((1, 4, 0), device="cuda", dtype=torch.int32)
    assert q.is_contiguous() and q.data_ptr() % 16 != 0
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        module.decode(q, k, k, shared, ex, 9, 16)


@pytest.mark.parametrize("mode,tile", [(0, 16), (2, 8), (5, 16), (6, 16), (7, 16), (8, 16), (9, 16), (10, 16), (11, 16)])
def test_graph_replay_output(module, mode, tile):
    torch.manual_seed(12)
    q = torch.randn((1, 4, 128), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 1, 8, 128), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    shared = torch.tensor([[[1, 2, 3]]], device="cuda", dtype=torch.int32)
    ex = torch.tensor([[[4], [5], [6], [7]]], device="cuda", dtype=torch.int32)
    mask = torch.zeros((1, 4, 8), device="cuda", dtype=torch.bool)
    mask[:, :, 1:4] = True
    for h in range(4):
        mask[0, h, 4 + h] = True
    expected = reference(q, k, v, mask)
    for _ in range(3):
        module.decode(q, k, v, shared, ex, mode, tile)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            output = module.decode(q, k, v, shared, ex, mode, tile)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, atol=0.003, rtol=0.003)
