import pytest

from summarize_qasper_component_projection import METHODS, combine


def test_projection_uses_measured_layers_and_dense_estimate():
    attn = []
    for layer in range(2, 32):
        for method in METHODS:
            duration = 4 if method == "fa2_dense" else 2 if method == "fi_dense" else 1
            attn.append({"step": 1, "layer": layer, "batch": 32, "method": method, "median_us": duration})
    common = [{"layer": layer, "batch": 32, "median_us": 10} for layer in range(32)]
    common.append({"layer": "output", "batch": 32, "median_us": 20})
    result = {row["method"]: row for row in combine(attn, common, [32], [1])}
    assert result["inline8"]["frontend_excluded_step_projection_ms"] == pytest.approx(0.374)
    assert result["fi_dense"]["frontend_excluded_step_projection_ms"] == pytest.approx(0.404)
    assert result["inline8"]["speedup_vs_fi_dense"] == pytest.approx(404 / 374)
    assert result["inline8"]["selection_cache_metadata_ms"] == "excluded"

    less = [{"step": 1, "layer": layer, "batch": 32, "method": method, "median_us": 3}
            for layer in range(3, 32) if layer != 13 for method in ("less_patched_headpacked", "less_fi_adapter")]
    with_less = {row["method"]: row for row in combine(attn, common, [32], [1], less)}
    # 28 sparse layers at 3 us, two dense search layers at 2 us, then
    # the first two layers estimated at 2 us each.
    assert with_less["less_patched_headpacked"]["frontend_excluded_step_projection_ms"] == pytest.approx(0.432)


def test_rejects_duplicate_rows():
    duplicate = {"step": 1, "layer": 2, "batch": 32, "method": "inline8", "median_us": 1}
    with pytest.raises(ValueError, match="duplicate"):
        combine([duplicate, duplicate], [], [32], [1])
