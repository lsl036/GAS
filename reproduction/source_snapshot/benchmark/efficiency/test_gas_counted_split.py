"""Verify split counted attention against heterogeneous FP32 references."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from test_gas_counted_decode import test_counted_reference as reference_case
from test_gas_counted_decode import test_graph_reads_updated_counts as graph_case

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def modules():
    split = load(name="gas_counted_split", sources=[str(ROOT / "csrc/src/gas_counted_split.cu")],
                 extra_cuda_cflags=["-O3"], verbose=False)
    counted = load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")],
                   extra_cuda_cflags=["-O3"], verbose=False)
    return split, counted


def adapter(modules, parts):
    split, counted = modules

    class Adapter:
        count_prefix = staticmethod(counted.count_prefix)

        def decode(self, q, k, v, shared, exceptions, sc, ec, segments):
            return split.decode(q, k, v, shared, exceptions, sc, ec, segments, parts)

    return Adapter()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("segments", [8, 16])
@pytest.mark.parametrize("parts", [2, 4, 8])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
def test_split_reference(modules, dtype, segments, parts, case):
    reference_case(adapter(modules, parts), dtype, segments, case, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("parts", [2, 4, 8])
def test_split_graph(modules, dtype, parts):
    graph_case(adapter(modules, parts), dtype, False)
