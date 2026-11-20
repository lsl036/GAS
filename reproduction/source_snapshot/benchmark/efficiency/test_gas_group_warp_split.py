"""Check warp-cooperative group split against heterogeneous references."""
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
    warp = load(name="gas_group_warp_split", sources=[str(ROOT / "csrc/src/gas_group_warp_split.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)
    return counted, warp


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
@pytest.mark.parametrize("segments,parts,chunk", [(4, 4, 128), (4, 8, 128), (4, 16, 64), (4, 32, 16), (8, 16, 64), (8, 8, 2048)])
@pytest.mark.parametrize("tiled", [False, True])
def test_reference(modules, dtype, case, segments, parts, chunk, tiled):
    counted, warp = modules

    class Adapter:
        count_prefix = staticmethod(counted.count_prefix)

        @staticmethod
        def decode(q, k, v, shared, ex, sc, ec, _segments):
            call = warp.decode_tiled if tiled else warp.decode
            return call(q, k, v, shared, ex, sc, ec, segments, parts, chunk)

    counted_reference(Adapter(), dtype, segments, case, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tiled", [False, True])
def test_dynamic_graph(modules, dtype, tiled):
    _, warp = modules

    class Adapter:
        @staticmethod
        def decode(q, k, v, shared, ex, sc, ec, chunk, max_parts, _blocks):
            call = warp.decode_tiled if tiled else warp.decode
            return call(q, k, v, shared, ex, sc, ec, 4, max_parts, chunk)

    graph_reference((None, Adapter()), dtype)
