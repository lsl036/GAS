"""Check the two-pass sparse decode against a dense attention reference."""
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
    return load(name="gas_twopass_decode_guarded", sources=[str(ROOT / "csrc/src/gas_twopass_decode.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["uneven", "no_shared", "no_exceptions"])
@pytest.mark.parametrize("segments", [4, 8, 16, 32])
@pytest.mark.parametrize("inline", [False, True])
def test_reference(module, dtype, case, segments, inline):
    counted_reference(module, dtype, segments, case, inline)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inline", [False, True])
def test_graph(module, dtype, inline):
    graph_reference(module, dtype, inline)


def test_scratch_limit(module):
    q = torch.zeros((1, 4, 128), device="cuda", dtype=torch.float16)
    k = torch.zeros((1, 1, 1, 128), device="cuda", dtype=torch.float16)
    shared = torch.zeros((1, 1, 8192), device="cuda", dtype=torch.int32)
    exceptions = torch.empty((1, 4, 0), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="shared memory"):
        module.decode_prefix(q, k, k, shared, exceptions, 32)
