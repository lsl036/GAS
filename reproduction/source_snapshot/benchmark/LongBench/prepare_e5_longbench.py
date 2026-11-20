#!/usr/bin/env python3
"""Freeze the E5 LongBench task/sample protocol and normalize reusable runs."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LONGBENCH = ROOT / "benchmark" / "LongBench"
DEFAULT_DATA = Path("/data/home/lsl/SparseAtt/data/LongBench/data")
DEFAULT_OUTPUT = ROOT / "benchmark" / "results_longbench" / "e5_longbench_completion"
MODEL_PATH = Path("/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct")
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"

TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "triviaqa",
    "passage_retrieval_en",
    "lcc",
    "repobench",
]
DATASET_KEY = {"repobench": "repobench-p"}
REUSED = {
    "quest_twi": ROOT
    / "benchmark"
    / "results_longbench"
    / "Meta-Llama-3.1-8B-Instruct_quest_twi_ext"
    / "pred",
    "group_max": ROOT
    / "benchmark"
    / "results_longbench"
    / "Meta-Llama-3.1-8B-Instruct_quest_twi_group_max_p080_exception16_ext"
    / "pred",
}
METHOD_CONFIGS = {
    "full": ROOT / "benchmark" / "configs" / "config_full.json",
    "quest_4pct": ROOT / "benchmark" / "configs" / "config_quest_4pct.json",
    "quest_8pct": ROOT / "benchmark" / "configs" / "config_quest_8pct.json",
    "prob_sum": ROOT
    / "benchmark"
    / "configs"
    / "config_quest_twi_gqa_prob_sum_p080_exception16.json",
    "quest_twi": ROOT / "benchmark" / "configs" / "config_quest_twi.json",
    "group_max": ROOT
    / "benchmark"
    / "configs"
    / "config_quest_twi_gqa_group_max_p080_exception16.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sample_hash(row: dict) -> str:
    return hashlib.sha256(canonical_json(row)).hexdigest()


def dataset_key(task: str) -> str:
    return DATASET_KEY.get(task, task)


def read_dataset(data_dir: Path, task: str) -> list[dict]:
    path = data_dir / f"{dataset_key(task)}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LongBench data file: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"Empty LongBench data file: {path}")
    return rows


def find_prediction(pred_dir: Path, task: str) -> Path:
    prefix = dataset_key(task).split("-")[0]
    candidates = sorted(
        path
        for path in pred_dir.glob(f"{prefix}-*.jsonl")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected one legacy prediction for {task} in {pred_dir}, found {candidates}")
    return candidates[0]


def normalize_reused(
    method: str, source_dir: Path, rows_by_task: dict[str, list[dict]], output_dir: Path
) -> dict:
    source_paths = {}
    for task, rows in rows_by_task.items():
        source = find_prediction(source_dir, task)
        with source.open("r", encoding="utf-8") as f:
            predictions = [json.loads(line) for line in f if line.strip()]
        if len(predictions) != len(rows):
            raise ValueError(
                f"Legacy {method}/{task} count mismatch: {len(predictions)} != {len(rows)} ({source})"
            )
        normalized = []
        for index, (row, prediction) in enumerate(zip(rows, predictions)):
            for field in ("answers", "all_classes", "length"):
                if prediction.get(field) != row.get(field):
                    raise ValueError(
                        f"Legacy {method}/{task} row {index} differs in {field}; "
                        "refusing to reuse a different sample order"
                    )
            normalized.append(
                {
                    **prediction,
                    "_e5_sample_id": f"{task}:{index:06d}",
                    "_e5_sample_hash": sample_hash(row),
                    "_e5_row_index": index,
                    "_e5_reused": True,
                    "_e5_source_path": str(source.resolve()),
                }
            )
        target = output_dir / method / task / "predictions.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            for prediction in normalized:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        temporary.replace(target)
        source_paths[task] = str(source.resolve())
    return source_paths


def command_text(command: list[str]) -> str:
    return " ".join(command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_task = {task: read_dataset(args.data_dir, task) for task in TASKS}

    manifest_path = output_dir / "manifest.jsonl"
    manifest_rows = []
    for task in TASKS:
        for index, row in enumerate(rows_by_task[task]):
            manifest_rows.append(
                {
                    "task": task,
                    "dataset_key": dataset_key(task),
                    "row_index": index,
                    "sample_id": f"{task}:{index:06d}",
                    "sample_hash": sample_hash(row),
                    "length": row.get("length"),
                }
            )
    temporary_manifest = manifest_path.with_suffix(".tmp")
    with temporary_manifest.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_manifest.replace(manifest_path)

    dataset_config = LONGBENCH / "config"
    prompt_config = dataset_config / "dataset2prompt.json"
    maxlen_config = dataset_config / "dataset2maxlen.json"
    prompt_data = json.loads(prompt_config.read_text(encoding="utf-8"))
    maxlen_data = json.loads(maxlen_config.read_text(encoding="utf-8"))
    for task in TASKS:
        key = dataset_key(task)
        if key not in prompt_data or key not in maxlen_data:
            raise ValueError(f"Missing prompt/max-generation configuration for {task} ({key})")

    reused_sources = {
        method: normalize_reused(method, source, rows_by_task, output_dir / "pred")
        for method, source in REUSED.items()
    }
    config_records = {}
    for method, config_path in METHOD_CONFIGS.items():
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config_records[method] = {
            "path": str(config_path.resolve()),
            "sha256": sha256(config_path),
            "config": json.loads(config_path.read_text(encoding="utf-8")),
            "reused": method in reused_sources,
        }

    protocol = {
        "protocol": "E5 LongBench completion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_short": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "model": {
            "name": MODEL_NAME,
            "checkpoint": str(args.model_path.resolve()),
            "max_input_tokens": 31500,
        },
        "tasks": TASKS,
        "task_mapping": DATASET_KEY,
        "sample_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "total_samples": len(manifest_rows),
            "counts": {task: len(rows_by_task[task]) for task in TASKS},
            "hash": "sha256(canonical JSONL row)",
        },
        "dataset": {
            "directory": str(args.data_dir.resolve()),
            "files": {
                dataset_key(task): sha256(args.data_dir / f"{dataset_key(task)}.jsonl")
                for task in TASKS
            },
            "split": "local LongBench test files",
        },
        "prompt": {
            "config": str(prompt_config.resolve()),
            "config_sha256": sha256(prompt_config),
            "max_generation_config": str(maxlen_config.resolve()),
            "max_generation_config_sha256": sha256(maxlen_config),
            "truncation": "middle truncation to 31500 tokens when needed",
            "chat_template": "benchmark/LongBench/pred.py build_chat for Llama-3.1",
        },
        "generation": {
            "dtype": "bfloat16",
            "do_sample": False,
            "num_beams": 1,
            "seed": 42,
            "past_key_value_decode": True,
            "output_postprocess": "LongBench pred.py post_process",
        },
        "methods": config_records,
        "reused_sources": reused_sources,
        "new_methods": ["full", "quest_4pct", "quest_8pct", "prob_sum"],
        "evaluation": {
            "metric": "task-specific benchmark/LongBench/metrics.py",
            "macro": "equal mean of 13 unrounded task scores",
            "empty_prediction": "scored by the original evaluator, not dropped",
        },
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": __import__("torch").__version__,
        },
        "normalization": {
            "legacy_predictions": "enriched with row-index sample IDs and hashes after answers/length checks",
            "new_predictions": "checkpointed by benchmark/LongBench/pred.py",
        },
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest_rows)} manifest rows to {manifest_path}")
    print(f"wrote protocol to {protocol_path}")
    for method, paths in reused_sources.items():
        print(f"normalized {method}: {len(paths)} tasks")


if __name__ == "__main__":
    main()
