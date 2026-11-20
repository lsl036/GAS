"""GPU correctness for the patched experimental head-packed LessIsMore adapter."""
from pathlib import Path

import pytest
import torch

from bench_e6_qasper_official_b1 import load_extension
from bench_e6_qasper_trace import reference
from bench_less_official_headpacked import official_grouped_setup

MODULE = Path("/data/tmp/torch_ext_e6_stream_adapters/e6_less_nopart_fixed/e6_less_nopart_fixed.so")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not MODULE.exists(), reason="CUDA and built adapter required")


@pytest.fixture(scope="module")
def module():
    return load_extension(MODULE, "e6_less_nopart_fixed")


@pytest.mark.parametrize("batch", [32, 128])
@pytest.mark.parametrize("budget", [2, 17, 128, 257])
def test_reference_and_graph(module, batch, budget):
    torch.manual_seed(13 + batch + budget)
    length = 384
    q = torch.randn(batch, 32, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, 8, length, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    row = torch.cat((torch.arange(budget - 1, device="cuda"), torch.tensor([length - 1], device="cuda")))
    ids = row.expand(batch, -1).contiguous()
    mask = torch.zeros(batch, 32, length, device="cuda", dtype=torch.bool)
    mask.scatter_(2, ids[:, None, :].expand(batch, 32, budget), True)
    call, wrappers = official_grouped_setup(module, q, k, v, ids, [length] * batch, batch)
    try:
        expected = reference(q, k, v, mask)
        output = call()
        torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
        q.add_(0.1)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        expected = reference(q, k, v, mask)
        torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
    finally:
        for wrapper in wrappers:
            wrapper.end_forward()
