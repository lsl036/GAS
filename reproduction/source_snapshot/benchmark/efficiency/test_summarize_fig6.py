import csv

import pytest

from summarize_fig6 import BATCHES, LESS_LAYERS, read_rows


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("layer", "batch", "method", "samples", "median_us"))
        writer.writeheader()
        writer.writerows(rows)


def test_strict_coverage_and_sample_count(tmp_path):
    path = tmp_path / "results.csv"
    rows = [{"layer": layer, "batch": batch, "method": "less_fi_adapter", "samples": 20, "median_us": 1}
            for layer in LESS_LAYERS for batch in BATCHES]
    write_rows(path, rows)
    assert len(read_rows(path, "less_fi_adapter", LESS_LAYERS)) == 112
    write_rows(path, rows[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        read_rows(path, "less_fi_adapter", LESS_LAYERS)
    rows[0]["samples"] = 19
    write_rows(path, rows)
    with pytest.raises(ValueError, match="sample count"):
        read_rows(path, "less_fi_adapter", LESS_LAYERS)
    rows[0]["samples"] = 20
    rows[0]["median_us"] = "nan"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="invalid kernel latency"):
        read_rows(path, "less_fi_adapter", LESS_LAYERS)
