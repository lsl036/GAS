"""Test heterogeneous counted supports, dtype and graph replay."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from bench_e6_qasper_trace import reference

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def module():
    return load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("segments", [4, 8, 16, 32])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
def test_counted_reference(module, dtype, segments, case, inline):
    torch.manual_seed(31)
    batch, groups, heads, length = 2, 8, 32, 1024
    q = torch.randn(batch, heads, 128, device="cuda", dtype=dtype)
    k = torch.randn(batch, groups, length, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    sw = 0 if case == "no_shared" else 960
    ew = 0 if case == "no_exceptions" else 16
    shared = torch.full((batch, groups, sw), -1, dtype=torch.int32)
    exceptions = torch.full((batch, heads, ew), -1, dtype=torch.int32)
    sc = torch.zeros((batch, groups), dtype=torch.int32)
    ec = torch.zeros((batch, heads), dtype=torch.int32)
    mask = torch.zeros((batch, heads, length), dtype=torch.bool)
    generator = torch.Generator().manual_seed(91)
    counts = [[3, 2, 2, 3, 2, 944, 2, 2], [7, 65, 1, 83, 5, 11, 37, 4]]
    for b in range(batch):
        for group in range(groups):
            permutation = torch.randperm(length, generator=generator)
            n = 0 if not sw else counts[b][group]
            selected = permutation[:n]
            shared[b, group, :n] = selected.to(torch.int32)
            sc[b, group] = n
            for h in range(group * 4, group * 4 + 4):
                count = 0 if not ew else 1 + ((h + b) % 5)
                extra = permutation[n:n + count]
                exceptions[b, h, :count] = extra.to(torch.int32)
                ec[b, h] = count
                mask[b, h, selected] = True
                mask[b, h, extra] = True
    shared, exceptions, sc, ec, mask = [x.to("cuda") for x in (shared, exceptions, sc, ec, mask)]
    prefix_sc, prefix_ec = module.count_prefix(shared, exceptions)
    assert torch.equal(prefix_sc, sc) and torch.equal(prefix_ec, ec)
    output = (module.decode_prefix(q, k, v, shared, exceptions, segments) if inline else
              module.decode(q, k, v, shared, exceptions, sc, ec, segments))
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    expected = reference(q, k, v, mask)
    torch.testing.assert_close(output.float(), expected, atol=0.012 if dtype == torch.bfloat16 else 0.003,
                               rtol=0.012 if dtype == torch.bfloat16 else 0.003)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_graph_reads_updated_counts(module, dtype, inline):
    torch.manual_seed(19)
    q = torch.randn((1, 4, 128), device="cuda", dtype=dtype)
    k = torch.randn((1, 1, 8, 128), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    shared = torch.tensor([[[1, 2, 3]]], device="cuda", dtype=torch.int32)
    exceptions = torch.tensor([[[4], [5], [6], [7]]], device="cuda", dtype=torch.int32)
    sc = torch.tensor([[3]], device="cuda", dtype=torch.int32)
    ec = torch.ones((1, 4), device="cuda", dtype=torch.int32)
    def call():
        return (module.decode_prefix(q, k, v, shared, exceptions, 8) if inline else
                module.decode(q, k, v, shared, exceptions, sc, ec, 8))
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    sc.fill_(1)
    shared[:, :, 1:].fill_(-1)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    mask = torch.zeros((1, 4, 8), dtype=torch.bool, device="cuda")
    mask[:, :, 1] = True
    for h in range(4):
        mask[0, h, 4 + h] = True
    expected = reference(q, k, v, mask)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected, atol=0.012 if dtype == torch.bfloat16 else 0.003,
                               rtol=0.012 if dtype == torch.bfloat16 else 0.003)


class TreeAdapter:
    def __init__(self, module):
        self.module = module

    def count_prefix(self, shared, exceptions):
        return self.module.count_prefix(shared, exceptions)

    def decode_prefix(self, q, k, v, shared, exceptions, segments):
        return self.module.decode_prefix_tree(q, k, v, shared, exceptions, segments)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("segments", [4, 8, 16, 32])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
def test_tree_reference(module, dtype, segments, case):
    test_counted_reference(TreeAdapter(module), dtype, segments, case, True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tree_graph_reads_updated_counts(module, dtype):
    test_graph_reads_updated_counts(TreeAdapter(module), dtype, True)


class AsyncAdapter(TreeAdapter):
    def decode_prefix(self, q, k, v, shared, exceptions, segments):
        return self.module.decode_prefix_async(q, k, v, shared, exceptions, segments)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("segments", [4, 8, 16])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
def test_async_reference(module, dtype, segments, case):
    test_counted_reference(AsyncAdapter(module), dtype, segments, case, True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_async_graph_reads_updated_counts(module, dtype):
    test_graph_reads_updated_counts(AsyncAdapter(module), dtype, True)
