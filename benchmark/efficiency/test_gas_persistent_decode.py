"""Correctness tests for device-side ragged scheduling."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from bench_e6_qasper_trace import reference
from test_gas_counted_decode import test_counted_reference as counted_reference

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def modules():
    counted = load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")],
                   extra_cuda_cflags=["-O3"], verbose=False)
    persistent = load(name="gas_persistent_decode", sources=[str(ROOT / "csrc/src/gas_persistent_decode.cu")],
                      extra_cuda_cflags=["-O3"], verbose=False)
    return counted, persistent


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
@pytest.mark.parametrize("chunk,blocks", [(16, 1), (128, 3), (2048, 256)])
def test_reference(modules, dtype, case, chunk, blocks):
    counted, persistent = modules

    class Adapter:
        count_prefix = staticmethod(counted.count_prefix)

        @staticmethod
        def decode(q, k, v, shared, ex, sc, ec, segments):
            return persistent.decode(q, k, v, shared, ex, sc, ec, chunk, 8, blocks)

    counted_reference(Adapter(), dtype, 8, case, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_graph_changes_partition_count(modules, dtype):
    _, persistent = modules
    torch.manual_seed(42)
    q = torch.randn(1, 4, 128, device="cuda", dtype=dtype)
    k = torch.randn(1, 1, 32, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    shared = torch.arange(32, device="cuda", dtype=torch.int32).reshape(1, 1, 32)
    exceptions = torch.empty(1, 4, 0, device="cuda", dtype=torch.int32)
    sc = torch.tensor([[2]], device="cuda", dtype=torch.int32)
    ec = torch.zeros(1, 4, device="cuda", dtype=torch.int32)
    for _ in range(3):
        persistent.decode(q, k, v, shared, exceptions, sc, ec, 8, 8, 16)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = persistent.decode(q, k, v, shared, exceptions, sc, ec, 8, 8, 16)
    for count in (20, 1, 32, 2):
        sc.fill_(count)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        mask = torch.arange(32, device="cuda").reshape(1, 1, 32).expand(1, 4, 32) < count
        expected = reference(q, k, v, mask)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), expected, atol=0.012 if dtype == torch.bfloat16 else 0.003,
                                   rtol=0.012 if dtype == torch.bfloat16 else 0.003)
