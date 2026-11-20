"""Validate group-cooperative split attention and graph replay."""
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
    split = load(name="gas_cooperative_split", sources=[str(ROOT / "csrc/src/gas_cooperative_split.cu")],
                 extra_cuda_cflags=["-O3"], verbose=False)
    return counted, split


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
@pytest.mark.parametrize("segments,chunk", [(4, 16), (8, 128), (16, 2048)])
@pytest.mark.parametrize("parts", [8, 16, 32])
def test_reference(modules, dtype, case, segments, chunk, parts):
    counted, split = modules

    class Adapter:
        count_prefix = staticmethod(counted.count_prefix)

        @staticmethod
        def decode(q, k, v, shared, ex, sc, ec, segments):
            return split.decode(q, k, v, shared, ex, sc, ec, segments, parts, chunk)

    counted_reference(Adapter(), dtype, segments, case, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("parts", [8, 16, 32])
def test_dynamic_graph(modules, dtype, parts):
    _, split = modules

    class Adapter:
        @staticmethod
        def decode(q, k, v, shared, ex, sc, ec, chunk, max_parts, blocks):
            return split.decode(q, k, v, shared, ex, sc, ec, 8, parts, chunk)

    graph_reference((None, Adapter()), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
@pytest.mark.parametrize("parts,chunk,blocks", [(8, 128, 17), (16, 64, 37), (32, 16, 217)])
def test_compact_reference(modules, dtype, case, parts, chunk, blocks):
    counted, split = modules

    class Adapter:
        count_prefix = staticmethod(counted.count_prefix)

        @staticmethod
        def decode(q, k, v, shared, ex, sc, ec, segments):
            return split.decode_compact(q, k, v, shared, ex, sc, ec, segments, parts, chunk, blocks)

    counted_reference(Adapter(), dtype, 8, case, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compact_dynamic_graph(modules, dtype):
    _, split = modules

    class Adapter:
        @staticmethod
        def decode(q, k, v, shared, ex, sc, ec, chunk, max_parts, blocks):
            return split.decode_compact(q, k, v, shared, ex, sc, ec, 8, max_parts, chunk, 17)

    graph_reference((None, Adapter()), dtype)
