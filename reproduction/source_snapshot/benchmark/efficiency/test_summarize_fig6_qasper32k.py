import json

import summarize_fig6_qasper32k as summary


def test_repeated_report_is_idempotent_and_labels_datasets(tmp_path, monkeypatch):
    output = tmp_path / "handoff.md"
    output.write_text("# Existing\n", encoding="utf-8")
    meta = tmp_path / "fig6_traces/qasper32k_repeated.jsonl.meta.json"
    meta.parent.mkdir()
    meta.write_text(json.dumps({"source_sha256": "source", "output_sha256": "output"}))

    def condition(label):
        value = 32.0 if "32k" in label else 16.0
        means = {(method, batch): value for method in summary.METHODS for batch in summary.BATCHES}
        support = {field: "1" for field in ("twi_per_head_tokens_mean", "twi_group_union_mean", "gas_per_head_tokens_mean", "gas_group_union_mean")}
        columns = {"Quest FI": {(2, 32): {"lengths": "[1, 2]"}},
                   "Twilight FI": {(layer, 32): support for layer in summary.LAYERS}}
        return means, means, {"K": 100, "mean_union": 100.0}, {}, columns

    monkeypatch.setattr(summary, "OUTPUT", output)
    monkeypatch.setattr(summary, "RESULTS", tmp_path)
    monkeypatch.setattr(summary, "load_condition", condition)
    summary.main()
    first = output.read_text()
    summary.main()
    assert output.read_text() == first
    assert first.count("## Additional: Qasper 16K vs Repeated-Context Qasper 32K") == 1
    assert "| 32 | 32K RULER |" in first
    assert "[16.0, 32.0]" in first
