"""GPU correctness for patched Quest grouped head-packed page attention."""
from pathlib import Path

import pytest
import torch

from bench_e6_qasper_official_b1 import load_extension
from bench_e6_qasper_trace import reference
from bench_quest_official_headpacked import quest_setup

MODULE = Path("/data/tmp/torch_ext_e6_stream_adapters/e6_quest_nopart_fixed/e6_quest_nopart_fixed.so")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not MODULE.exists(), reason="CUDA and built adapter required")


@pytest.fixture(scope="module")
def module():
    return load_extension(MODULE, "e6_quest_nopart_fixed")


@pytest.mark.parametrize("batch", [32, 64])
@pytest.mark.parametrize("full_pages", [2, 7])
@pytest.mark.parametrize("tail", [0, 5, 15])
def test_reference_and_graph(module, batch, full_pages, tail):
    torch.manual_seed(19 + batch + full_pages + tail)
    length = 128 + tail
    q = torch.randn(batch, 32, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, 8, length, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    mask = torch.zeros(batch, 32, length, device="cuda", dtype=torch.bool)
    mask[:, :, :full_pages * 16] = True
    if tail:
        mask[:, :, 128:] = True
    call, wrappers = quest_setup(module, q, k, v, mask, [length] * batch, 4096)
    try:
        expected = reference(q, k, v, mask)
        out = call()
        torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
    finally:
        for wrapper in wrappers:
            wrapper.end_forward()
