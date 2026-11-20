"""Compare cooperative GQA decode against dense masked attention."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from test_gas_counted_decode import test_counted_reference as counted_reference
from test_gas_counted_decode import test_graph_reads_updated_counts as graph_reference

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def module():
    counted = load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")],
                   extra_cuda_cflags=["-O3"], verbose=False)
    cooperative = load(name="gas_cooperative_decode", sources=[str(ROOT / "csrc/src/gas_cooperative_decode.cu")],
                       extra_cuda_cflags=["-O3"], verbose=False)

    class Adapter:
        count_prefix = staticmethod(counted.count_prefix)
        decode = staticmethod(cooperative.decode)

    return Adapter()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
@pytest.mark.parametrize("segments", [4, 8, 16])
def test_reference(module, dtype, case, segments):
    counted_reference(module, dtype, segments, case, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_graph(module, dtype):
    graph_reference(module, dtype, False)
