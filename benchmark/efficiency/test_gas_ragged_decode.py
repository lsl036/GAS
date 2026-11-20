"""Numerical and plan checks for ragged counted-support decoding."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from bench_e6_qasper_trace import reference
from gas_ragged_plan import build_plan, build_plan_gpu
from test_gas_counted_decode import test_counted_reference as reference_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def modules():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ragged = load(name="gas_ragged_decode", sources=[str(ROOT / "csrc/src/gas_ragged_decode.cu")], extra_cuda_cflags=["-O3"], verbose=False)
    counted = load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")], extra_cuda_cflags=["-O3"], verbose=False)
    return ragged, counted


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("segments", [4, 8, 16])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
@pytest.mark.parametrize("planner", ["cpu", "gpu"])
def test_reference(modules, dtype, segments, case, planner):
    ragged, counted = modules

    class Adapter:
        count_prefix = staticmethod(counted.count_prefix)

        def decode(self, q, k, v, shared, exceptions, sc, ec, n):
            plan = (build_plan(sc, ec, shared.shape[-1], exceptions.shape[-1], 128, 8, q.device) if planner == "cpu"
                    else build_plan_gpu(sc, ec, 128, 8))
            return ragged.decode(q, k, v, shared, exceptions, sc, ec, plan["jobs"], plan["parts"],
                                 plan["offsets"], plan["long_rows"], n, plan["partial_count"])

    reference_case(Adapter(), dtype, segments, case, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunk", [2, 128])
@pytest.mark.parametrize("planner", ["cpu", "gpu"])
def test_graph_output(modules, dtype, chunk, planner):
    ragged, _ = modules
    torch.manual_seed(42)
    q = torch.randn(1, 4, 128, device="cuda", dtype=dtype)
    k = torch.randn(1, 1, 8, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    shared = torch.tensor([[[1, 2, 3]]], device="cuda", dtype=torch.int32)
    ex = torch.tensor([[[4], [5], [6], [7]]], device="cuda", dtype=torch.int32)
    sc = torch.tensor([[3]], device="cuda", dtype=torch.int32)
    ec = torch.ones((1, 4), device="cuda", dtype=torch.int32)
    plan = build_plan(sc, ec, 3, 1, chunk, 8, q.device) if planner == "cpu" else build_plan_gpu(sc, ec, chunk, 8)
    def call():
        return ragged.decode(q, k, v, shared, ex, sc, ec, plan["jobs"], plan["parts"],
                             plan["offsets"], plan["long_rows"], 8, plan["partial_count"])
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    q.mul_(0.5)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    mask = torch.zeros((1, 4, 8), dtype=torch.bool, device="cuda")
    mask[:, :, 1:4] = True
    for h in range(4):
        mask[0, h, 4 + h] = True
    expected = reference(q, k, v, mask)
    torch.testing.assert_close(output.float(), expected, atol=0.012 if dtype == torch.bfloat16 else 0.003,
                               rtol=0.012 if dtype == torch.bfloat16 else 0.003)


def test_plan_coverage():
    sc = torch.tensor([[3, 944]], dtype=torch.int32)
    ec = torch.tensor([[0, 1, 0, 2, 0, 0, 16, 0]], dtype=torch.int32)
    plan = build_plan(sc, ec, 944, 16, 128, 8, torch.device("cpu"))
    assert plan["parts"].tolist() == [[1, 1, 1, 1, 8, 8, 8, 8]]
    assert plan["long_rows"].tolist() == [4, 5, 6, 7]
    assert plan["partial_count"] == 32
    jobs = plan["jobs"].tolist()
    assert len(jobs) == 36 and len({tuple(x) for x in jobs}) == 36
    for row in range(8):
        assert [p for h, p in jobs if h == row] == list(range(int(plan["parts"][0, row])))


def test_plan_rejects_invalid_counts():
    sc = torch.tensor([[4]], dtype=torch.int32)
    ec = torch.ones((1, 4), dtype=torch.int32)
    with pytest.raises(ValueError, match="out of range"):
        build_plan(sc, ec, 3, 1, 128, 8, torch.device("cpu"))
    with pytest.raises(ValueError, match="nonempty"):
        build_plan(torch.zeros_like(sc), torch.zeros_like(ec), 3, 1, 128, 8, torch.device("cpu"))


@pytest.mark.parametrize("chunk", [1, 128, 2048])
def test_gpu_plan_matches_cpu(modules, chunk):
    sc = torch.tensor([[3, 944], [0, 17]], dtype=torch.int32)
    ec = torch.tensor([[0, 1, 0, 2, 0, 0, 16, 0], [1, 2, 3, 4, 0, 2, 4, 6]], dtype=torch.int32)
    expected = build_plan(sc, ec, 944, 16, chunk, 8, torch.device("cpu"))
    actual = build_plan_gpu(sc.cuda(), ec.cuda(), chunk, 8)
    for name in ("jobs", "parts", "offsets", "long_rows"):
        assert torch.equal(expected[name], actual[name].cpu())
    assert expected["partial_count"] == actual["partial_count"]
