"""Audit existing RULER and LongBench prediction directories without rerunning models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summary_scores(path: Path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    task_row = next((row for row in rows if row and row[0] == "Tasks"), None)
    score_row = next((row for row in rows if row and row[0] == "Score"), None)
    if task_row is None or score_row is None:
        return {}
    return {
        task: score
        for task, score in zip(task_row[1:], score_row[1:])
        if task and score
    }


def longbench_scores(path: Path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        raw_scores = json.load(handle)
    return {
        filename.split("-{", 1)[0]: score
        for filename, score in raw_scores.items()
    }


def sample_hash(record):
    identity = {
        "input": record.get("input"),
        "outputs": record.get("outputs", record.get("output")),
        "others": record.get("others", {}),
    }
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inspect_prediction_dir(
    rows,
    benchmark: str,
    length: str,
    method: str,
    directory: Path,
):
    scores = (
        summary_scores(directory / "summary.csv")
        if benchmark == "ruler"
        else longbench_scores(directory / "result.json")
    )
    for prediction_path in sorted(directory.glob("*.jsonl")):
        if prediction_path.name.endswith(".profile.jsonl"):
            continue
        records = read_jsonl(prediction_path)
        indices = [record.get("index") for record in records]
        valid_indices = [index for index in indices if index is not None]
        sample_hashes = [sample_hash(record) for record in records]
        task = (
            prediction_path.stem
            if benchmark == "ruler"
            else prediction_path.stem.split("-{", 1)[0]
        )
        rows.append(
            {
                "benchmark": benchmark,
                "length": length,
                "method": method,
                "task": task,
                "prediction_count": len(records),
                "unique_index_value_count": len(set(valid_indices)),
                "duplicate_index_value_count": len(valid_indices) - len(set(valid_indices)),
                "unique_sample_hash_count": len(set(sample_hashes)),
                "duplicate_sample_hash_count": len(sample_hashes) - len(set(sample_hashes)),
                "sample_set_sha256": hashlib.sha256(
                    "\n".join(sorted(sample_hashes)).encode("ascii")
                ).hexdigest(),
                "score": scores.get(task, ""),
                "path": str(prediction_path),
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--longbench-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", default=["16384", "32768", "65536"])
    parser.add_argument("--expected-samples", type=int, default=50)
    args = parser.parse_args()

    inventory = []
    missing = []

    for length in args.lengths:
        length_root = args.ruler_root / str(length) / "pred"
        if not length_root.exists():
            for method in ("<missing-method-root>",):
                missing.append(
                    {
                        "benchmark": "ruler",
                        "length": length,
                        "method": method,
                        "task": "<missing-pred-root>",
                        "reason": str(length_root),
                    }
                )
            continue
        for method_dir in sorted(path for path in length_root.iterdir() if path.is_dir()):
            inspect_prediction_dir(inventory, "ruler", length, method_dir.name, method_dir)
            for task in RULER_TASKS:
                prediction_path = method_dir / f"{task}.jsonl"
                if not prediction_path.exists():
                    missing.append(
                        {
                            "benchmark": "ruler",
                            "length": length,
                            "method": method_dir.name,
                            "task": task,
                            "reason": "file_missing",
                        }
                    )
                else:
                    count = len(read_jsonl(prediction_path))
                    if count != args.expected_samples:
                        missing.append(
                            {
                                "benchmark": "ruler",
                                "length": length,
                                "method": method_dir.name,
                                "task": task,
                                "reason": f"count_{count}_expected_{args.expected_samples}",
                            }
                        )

    for model_dir in sorted(path for path in args.longbench_root.iterdir() if path.is_dir()):
        for prediction_root in (model_dir / "pred", model_dir / "pred_e"):
            if prediction_root.exists():
                inspect_prediction_dir(
                    inventory,
                    "longbench",
                    "actual_length",
                    model_dir.name,
                    prediction_root,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "existing_results_inventory.csv"
    missing_path = args.output_dir / "missing_runs.csv"
    fields = [
        "benchmark",
        "length",
        "method",
        "task",
        "prediction_count",
        "unique_index_value_count",
        "duplicate_index_value_count",
        "unique_sample_hash_count",
        "duplicate_sample_hash_count",
        "sample_set_sha256",
        "score",
        "path",
    ]
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)

    with missing_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["benchmark", "length", "method", "task", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing)

    print(f"inventory: {inventory_path} ({len(inventory)} rows)")
    print(f"missing: {missing_path} ({len(missing)} rows)")


if __name__ == "__main__":
    main()
