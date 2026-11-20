#!/usr/bin/env python3
"""Run the two-stage Fig. 4(a) KV-matched quality experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np


TASKS = ("niah_single_1", "niah_multikey_3", "niah_multivalue", "vt", "cwe")
LENGTH = 32768
CAL_SAMPLES = 10
FORMAL_SAMPLES = 50


def config_base() -> dict:
    return {
        "compressor": {"type": "none"},
        "weight_estimator": {"type": "none"},
        "weight_pruner": {"type": "none"},
        "skip_first_two_layers": True,
        "use_estimated_weights_in_attn": False,
    }


def quest_rate(rate: float) -> dict:
    value = config_base()
    value["selector"] = {
        "type": "quest", "token_budget": -1, "chunk_size": 16,
        "sparsity_rate": rate,
    }
    return value


def ds_rate(rate: float, channel_config: str) -> dict:
    value = config_base()
    value["selector"] = {
        "type": "ds", "quant_bit": 2, "r": 64,
        "config_path": channel_config, "token_budget": -1,
        "sparsity_rate": rate, "selected_channel": "q",
    }
    return value


def twilight(token_budget: int, threshold: float) -> dict:
    value = config_base()
    value["selector"] = {"type": "quest", "token_budget": token_budget, "chunk_size": 16}
    value["weight_estimator"] = {"type": "min_max_quant", "quant_bit": 4}
    value["weight_pruner"] = {"type": "top_p", "threshold": threshold}
    return value


def gas(selector: str, channel_config: str | None) -> dict:
    if selector == "quest":
        value = twilight(8192, 0.85)
    else:
        value = config_base()
        value["selector"] = {
            "type": "ds", "quant_bit": 2, "r": 64,
            "config_path": channel_config, "token_budget": 8192,
            "selected_channel": "q",
        }
        value["weight_estimator"] = {"type": "min_max_quant", "quant_bit": 4}
        value["weight_pruner"] = {"type": "top_p", "threshold": 0.85}
    value["gqa_refinement"] = {
        "enabled": True, "mode": "group_max", "threshold": 0.80,
        "exception_budget": 16,
    }
    return value


def write_config(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_summary(path: Path) -> dict[str, float]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    if len(rows) < 3 or rows[1][0] != "Tasks" or rows[2][0] != "Score":
        raise ValueError(f"Invalid RULER summary: {path}")
    return {task: float(score) for task, score in zip(rows[1][1:], rows[2][1:])}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sample_score(task: str, record: dict) -> float:
    prediction = str(record.get("pred", "")).strip().lower()
    refs = [str(ref).lower() for ref in record.get("outputs", [])]
    if not refs:
        return 0.0
    if task == "vt":
        return 100.0 * sum(ref in prediction for ref in refs) / len(refs)
    return 100.0 * sum(ref in prediction for ref in refs) / len(refs)


def method_dir(root: Path, model_name: str, phase: str, method: str) -> Path:
    return root / phase / model_name / "synthetic_e1" / str(LENGTH) / "pred" / method


def run_method(args, phase: str, method: str, config: Path, samples: int, seed: int) -> None:
    source_data = args.calibration_data_root if phase == "calibration" else args.formal_data_root
    if source_data:
        target_data = args.output_dir / phase / args.model_name / "synthetic_e1" / str(LENGTH) / "data"
        target_data.parent.mkdir(parents=True, exist_ok=True)
        if target_data.exists() and not target_data.is_symlink():
            raise RuntimeError(f"Refusing to overwrite existing data directory: {target_data}")
        if not target_data.exists():
            target_data.symlink_to(Path(source_data).resolve(), target_is_directory=True)
    output = method_dir(args.output_dir, args.model_name, phase, method)
    complete = all(
        (output / f"{task}.jsonl").is_file()
        and sum(1 for _ in (output / f"{task}.jsonl").open(encoding="utf-8")) == samples
        for task in TASKS
    )
    if complete:
        print(f"skip completed {phase}/{method}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / "logs" / f"{phase}_{method}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "TWILIGHT_MODEL_DIR": args.model_path,
        "TWILIGHT_MODEL_NAME": args.model_name,
        "TWILIGHT_RULER_ROOT_DIR": str(args.output_dir / phase),
        "TWILIGHT_SEQ_LENGTHS": str(LENGTH),
        "TWILIGHT_RULER_TASKS": " ".join(TASKS),
        "TWILIGHT_RULER_RANDOM_SEED": str(seed),
        "NUM_SAMPLES": str(samples),
        "TWILIGHT_RULER_PROFILE": "1",
        "TWILIGHT_RULER_PROFILE_RAW": "0",
    })
    command = ["bash", str(args.ruler_scripts / "run.sh"), str(config), method, "synthetic_e1"]
    print("RUN", phase, method, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=args.ruler_scripts, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def load_metrics(root: Path, model_name: str, phase: str, method: str, samples: int, require_kv: bool = True) -> tuple[dict, dict[str, list[float]]]:
    base = method_dir(root, model_name, phase, method)
    scores = read_summary(base / "summary.csv")
    loaded: dict[str, list[float]] = {}
    for task in TASKS:
        records = read_jsonl(base / f"{task}.jsonl")
        if len(records) != samples:
            raise ValueError(f"{phase}/{method}/{task}: expected {samples}, got {len(records)}")
        values = []
        for record in records:
            value = record.get("kv_metrics", {}).get("final_loaded_kv_tokens_mean")
            if not isinstance(value, (int, float)):
                if not require_kv:
                    continue
                raise ValueError(f"missing KV metric in {phase}/{method}/{task}")
            values.append(float(value))
        loaded[task] = values
    return scores, loaded


def macro(scores: dict[str, float]) -> float:
    return float(np.mean([scores[task] for task in TASKS]))


def mean_loaded(loaded: dict[str, list[float]]) -> float:
    return float(np.mean([np.mean(loaded[task]) for task in TASKS]))


def make_calibration_methods(args, config_dir: Path) -> dict[str, Path]:
    methods: dict[str, Path] = {}
    gas_id = "gas"
    gas_config = config_dir / f"{gas_id}.json"
    write_config(gas_config, gas(args.selector, args.channel_config))
    methods[gas_id] = gas_config
    for rate in (0.08, 0.04, 0.02, 0.01, 0.005, 0.0025):
        for selector in ("quest", "ds"):
            ident = f"{selector}_rate_{str(rate).replace('.', 'p')}"
            path = config_dir / f"{ident}.json"
            write_config(path, quest_rate(rate) if selector == "quest" else ds_rate(rate, args.channel_config))
            methods[ident] = path
    for threshold in (0.05, 0.10, 0.20, 0.25, 0.40, 0.60, 0.85, 1.0):
        ident = f"twi_pt_{str(threshold).replace('.', 'p')}"
        path = config_dir / f"{ident}.json"
        write_config(path, twilight(8192, threshold))
        methods[ident] = path
    for budget in (4096, 2048, 1024, 512, 256):
        ident = f"twi_k_{budget}"
        path = config_dir / f"{ident}.json"
        write_config(path, twilight(budget, 0.85))
        methods[ident] = path
    return methods


def select_calibrated(calibration: list[dict], target: float, family: str | None = None) -> dict:
    rows = [row for row in calibration if family is None or row["family"] == family]
    valid = [row for row in rows if 1.0 <= row["loaded_kv"] / target <= 1.05]
    if valid:
        return max(valid, key=lambda row: (row["score"], -abs(row["loaded_kv"] / target - 1.0)))
    upper = [row for row in rows if row["loaded_kv"] >= target]
    if upper:
        return min(upper, key=lambda row: row["loaded_kv"] / target)
    return min(rows, key=lambda row: abs(row["loaded_kv"] / target - 1.0))


def inherited_selection(
    methods: dict[str, Path], method_id: str, family: str, selector: str
) -> dict:
    return {
        "method_id": method_id,
        "family": family,
        "selector": selector,
        "score": None,
        "loaded_kv": None,
        "config": str(methods[method_id]),
        "selection_source": "inherited_from_llama31_fig4a",
    }


def bootstrap_diff(candidate: dict[str, list[float]], baseline: dict[str, list[float]], seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(10000):
        c_means = []
        b_means = []
        for task in TASKS:
            idx = rng.integers(0, len(candidate[task]), size=len(candidate[task]))
            c_means.append(float(np.mean(np.asarray(candidate[task])[idx])))
            b_means.append(float(np.mean(np.asarray(baseline[task])[idx])))
        draws.append(float(np.mean(c_means) - np.mean(b_means)))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--selector", choices=("quest", "ds"), required=True)
    parser.add_argument("--channel-config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-data-root", type=Path)
    parser.add_argument("--formal-data-root", type=Path)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--ruler-scripts", type=Path, default=Path(__file__).resolve().parents[1] / "scripts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_dir / "configs"
    calibration_methods = make_calibration_methods(args, config_dir)
    calibration_rows = []
    if args.skip_calibration:
        target = None
        selections = {
            "quest_cal": inherited_selection(calibration_methods, "quest_rate_0p0025", "quest", "quest"),
            "ds_cal": inherited_selection(calibration_methods, "ds_rate_0p0025", "ds", "ds"),
            "twi_cal": inherited_selection(calibration_methods, "twi_k_2048", "twilight", "quest"),
        }
        frozen = {
            "model": args.model_name,
            "context_length": LENGTH,
            "calibration_samples_per_task": 0,
            "formal_samples_per_task": FORMAL_SAMPLES,
            "target_loaded_kv": None,
            "selection_source": "inherited_from_llama31_fig4a",
            "selections": selections,
        }
    else:
        for method, config in calibration_methods.items():
            run_method(args, "calibration", method, config, CAL_SAMPLES, 42)
            scores, loaded = load_metrics(args.output_dir, args.model_name, "calibration", method, CAL_SAMPLES)
            selector = "gas" if method == "gas" else ("ds" if method.startswith("ds_") else ("quest" if method.startswith("quest_") else "quest"))
            family = "gas" if method == "gas" else ("quest" if method.startswith("quest_") else ("ds" if method.startswith("ds_") else "twilight"))
            calibration_rows.append({"method_id": method, "family": family, "selector": selector, "score": macro(scores), "loaded_kv": mean_loaded(loaded), "config": str(config)})
        target = next(row["loaded_kv"] for row in calibration_rows if row["method_id"] == "gas")
        selections = {
            "quest_cal": select_calibrated(calibration_rows, target, "quest"),
            "ds_cal": select_calibrated(calibration_rows, target, "ds"),
            "twi_cal": select_calibrated(calibration_rows, target, "twilight"),
        }
        frozen = {"model": args.model_name, "context_length": LENGTH, "calibration_samples_per_task": CAL_SAMPLES, "formal_samples_per_task": FORMAL_SAMPLES, "target_loaded_kv": target, "selections": selections}
    (args.output_dir / "frozen_selection.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    if calibration_rows:
        with (args.output_dir / "calibration.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(calibration_rows[0]) + ["ratio_to_target", "match_status"])
            writer.writeheader()
            for row in calibration_rows:
                row["ratio_to_target"] = row["loaded_kv"] / target
                row["match_status"] = "in_range" if 1.0 <= row["ratio_to_target"] <= 1.05 else "out_of_range"
                writer.writerow(row)

    formal_configs = {
        "full": args.ruler_scripts.parent.parent / "configs/config_full.json",
        "gas": calibration_methods["gas"],
        "quest_8pct": calibration_methods["quest_rate_0p08"],
        "ds_8pct": calibration_methods["ds_rate_0p08"],
        "twilight_default": calibration_methods["twi_pt_0p85"],
        "quest_cal": Path(selections["quest_cal"]["config"]),
        "ds_cal": Path(selections["ds_cal"]["config"]),
        "twilight_cal": Path(selections["twi_cal"]["config"]),
    }
    formal_rows = []
    sample_scores = {}
    sample_loaded = {}
    for method, config in formal_configs.items():
        run_method(args, "formal", method, config, FORMAL_SAMPLES, 1337)
        scores, loaded = load_metrics(args.output_dir, args.model_name, "formal", method, FORMAL_SAMPLES, require_kv=method != "full")
        sample_loaded[method] = loaded
        sample_scores[method] = {task: [sample_score(task, record) for record in read_jsonl(method_dir(args.output_dir, args.model_name, "formal", method) / f"{task}.jsonl")] for task in TASKS}
        selector = "ds" if method.startswith("ds") else "quest"
        strategy = "gas" if method == "gas" else ("twilight" if method.startswith("twilight") else "selector_only")
        budget_type = "fixed_8192" if method in ("gas", "twilight_default") else ("calibrated" if method.endswith("_cal") else "8pct")
        has_loaded_kv = any(values for values in loaded.values())
        selected_id = selections[method]["method_id"] if method in selections else method
        p_t = 0.85 if strategy in ("gas", "twilight") else ""
        if selected_id.startswith("twi_pt_"):
            p_t = float(selected_id.removeprefix("twi_pt_").replace("p", "."))
        budget_value = 8192 if method in ("gas", "twilight_default") else (0.08 if method in ("quest_8pct", "ds_8pct") else selected_id)
        formal_rows.append({"method_id": method, "selector": selector, "strategy": strategy, "budget_type": budget_type, "budget_value": budget_value, "effective_candidates": selected_id, "p_t": p_t, "p_g": 0.80 if strategy == "gas" else "", "m": 16 if strategy == "gas" else "", "n_samples": FORMAL_SAMPLES * len(TASKS), "macro_score": macro(scores), "mean_loaded_KV": mean_loaded(loaded) if has_loaded_kv else "", "KV_ratio_to_GAS": "", "match_status": "", "source_path": str(method_dir(args.output_dir, args.model_name, "formal", method))})
    gas_kv = next(row["mean_loaded_KV"] for row in formal_rows if row["method_id"] == "gas")
    for row in formal_rows:
        row["KV_ratio_to_GAS"] = row["mean_loaded_KV"] / gas_kv if row["mean_loaded_KV"] != "" else ""
        row["match_status"] = "reference" if row["method_id"] == "gas" else "formal"
    with (args.output_dir / "fig4a_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(formal_rows[0]))
        writer.writeheader()
        writer.writerows(formal_rows)
    gas_scores = sample_scores["gas"]
    pair_rows = []
    for method in formal_configs:
        if method == "gas":
            continue
        low, high = bootstrap_diff(sample_scores[method], gas_scores, 42)
        pair_rows.append({"method_id": method, "score_delta_vs_GAS": macro({task: np.mean(sample_scores[method][task]) - np.mean(gas_scores[task]) for task in TASKS}), "ci95_low": low, "ci95_high": high})
    with (args.output_dir / "pairwise_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    (args.output_dir / "Fig4a_Result_Summary.md").write_text(
        "# Fig. 4(a) KV-Matched Quality Summary\n\n"
        f"Model: `{args.model_name}`; context: 32K; tasks: {', '.join(TASKS)}.\n\n"
        f"Calibration target (GAS mean loaded KV): {target if target is not None else 'inherited parameters only'}. Formal GAS mean loaded KV: {gas_kv:.2f}.\n\n"
        "The formal table is in `fig4a_summary.csv`; paired score bootstrap intervals are in `pairwise_statistics.csv`.\n",
        encoding="utf-8",
    )
    print(f"Fig4a complete: {args.output_dir}")


if __name__ == "__main__":
    main()
