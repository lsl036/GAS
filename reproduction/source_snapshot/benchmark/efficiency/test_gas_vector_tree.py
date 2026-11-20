"""Exercise the separate tree-merge GAS kernel on the shared test fixtures."""
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from test_gas_warp_decode import test_graph_replay_output as graph_case
from test_gas_warp_decode import test_reference as reference_case

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def module():
    raw = load(name="gas_vector_tree", sources=[str(ROOT / "csrc/src/gas_vector_tree.cu")],
               extra_cuda_cflags=["-O3"], verbose=False)

    class Adapter:
        def decode(self, q, k, v, shared, exceptions, mode, _tile):
            return raw.decode(q, k, v, shared, exceptions, mode)

    return Adapter()


@pytest.mark.parametrize("segments", [8, 16, 32])
@pytest.mark.parametrize("case", ["mixed", "no_shared", "no_exceptions"])
def test_tree_reference(module, segments, case):
    reference_case(module, segments, 16, case)


@pytest.mark.parametrize("segments", [8, 16, 32])
def test_tree_graph(module, segments):
    graph_case(module, segments, 16)
