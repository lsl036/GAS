import csv
import json
import sys

import pytest

import summarize_fig6
import verify_fig6_main as verify


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_verify_main_requires_valid_raw_samples(tmp_path, monkeypatch):
    layers, batches, methods = (2, 3), (32, 64), ("fi_gas_adapter",)
    monkeypatch.setattr(verify, "LAYERS", layers)
    monkeypatch.setattr(verify, "BATCHES", batches)
    monkeypatch.setattr(verify, "METHODS", methods)
    monkeypatch.setattr(summarize_fig6, "BATCHES", batches)
    dirs = [tmp_path / "trace0", tmp_path / "trace1"]
    for index, directory in enumerate(dirs):
        directory.mkdir()
        (directory / "manifest.json").write_text(str(index))
        for layer in layers:
            (directory / f"step001_layer{layer:02d}.pt").write_bytes(f"{index}/{layer}".encode())
    base = tmp_path / "main"
    base.mkdir()
    config = {"manifest_sha256": [verify.digest(d / "manifest.json") for d in dirs],
              "script_sha256": verify.digest(verify.Path(verify.__file__).with_name("bench_gas_mixed_trace.py")),
              "steps": [1], "layers": list(layers), "batch_sizes": list(batches), "methods": list(methods),
              "dtype": "float16", "repeats": 10, "rounds": 2, "graph_iterations": 8}
    (base / "config.json").write_text(json.dumps(config))
    rows = []
    for layer in layers:
        hashes = [verify.digest(d / f"step001_layer{layer:02d}.pt") for d in dirs]
        for batch in batches:
            row = {"layer": str(layer), "batch": str(batch), "method": methods[0], "samples": "20",
                   "median_us": "1", "trace_sha256": json.dumps(hashes)}
            rows.append(row)
            write_csv(base / "shards" / f"s1_l{layer}_b{batch}_{methods[0]}.csv", [row])
            for round_id in range(2):
                write_csv(base / "samples" / f"s1_l{layer}_b{batch}_{methods[0]}_{round_id}.csv",
                          [{"latency_us": "1"}] * 10)
    write_csv(base / "results.csv", rows)
    monkeypatch.setattr(sys, "argv", ["verify_fig6_main.py", "--output-dir", str(base), "--trace-dirs", *map(str, dirs)])
    verify.main()
    write_csv(base / "samples" / f"s1_l2_b32_{methods[0]}_0.csv", [{"latency_us": "1"}])
    with pytest.raises(ValueError, match="missing raw timings"):
        verify.main()
