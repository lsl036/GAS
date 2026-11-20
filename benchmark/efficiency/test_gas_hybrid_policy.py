"""Check the fixed pilot-derived dispatch boundary."""
import pytest

from bench_gas_warp_replay import hybrid_backend


@pytest.mark.parametrize("batch,width,expected", [
    (1, 1, "flashinfer_exact"),
    (32, 95, "vector16_8"), (32, 96, "flashinfer_exact"),
    (64, 93, "vector16_8"), (64, 94, "flashinfer_exact"),
    (128, 93, "vector16_8"), (128, 94, "flashinfer_exact"),
])
def test_fixed_policy(batch, width, expected):
    assert hybrid_backend(batch, width) == expected
