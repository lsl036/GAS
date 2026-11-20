"""Validate dynamic split attention, including CUDA Graph replay."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from test_gas_counted_decode import test_counted_reference as counted_reference
from test_gas_persistent_decode import test_graph_changes_partition_count as graph_reference

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def modules():
    counted = load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")],
                   extra_cuda_cflags=["-O3"], verbose=False)
    adaptive = load(name="gas_adaptive_split", sources=[str(ROOT / "csrc/src/gas_adaptive_split.cu")],
                    extra_cuda_cflags=["-O3"], verbose=False)
    return counted, adaptive


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
@pytest.mark.parametrize("chunk", [16, 128, 2048])
def test_reference(modules, dtype, case, chunk):
    counted, adaptive = modules

    class Adapter:
        count_prefix = staticmethod(counted.count_prefix)

        @staticmethod
        def decode(q, k, v, shared, ex, sc, ec, segments):
            return adaptive.decode(q, k, v, shared, ex, sc, ec, segments, 8, chunk)

    counted_reference(Adapter(), dtype, 8, case, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dynamic_graph(modules, dtype):
    _, adaptive = modules

    class Adapter:
        @staticmethod
        def decode(q, k, v, shared, ex, sc, ec, chunk, max_parts, blocks):
            return adaptive.decode(q, k, v, shared, ex, sc, ec, 8, max_parts, chunk)

    graph_reference((None, Adapter()), dtype)
