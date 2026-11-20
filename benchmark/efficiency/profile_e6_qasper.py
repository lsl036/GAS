#!/usr/bin/env python3
"""Profile one real Qasper sample; this is not an attention kernel benchmark."""

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/LongBench"))

CONFIGS = {
    "quest": "config_quest_8pct.json",
    "quest_twi": "config_quest_twi_ratio8_gqa_metrics.json",
    "gas": "config_quest_twi_gas_ratio8_llama31.json",
}


def summarize(records):
    sparse = [r for r in records if r["layer_id"] >= 2]
    if not sparse:
        raise ValueError("No sparse-layer decode records were produced")
    grouped = defaultdict(list)
    for record in sparse:
        grouped[record["layer_id"]].append(record)
    if set(grouped) != set(range(2, 32)):
        raise ValueError(f"Expected all 30 sparse layers, got {sorted(grouped)}")
    if len({len(rows) for rows in grouped.values()}) != 1:
        raise ValueError("Unequal decode step counts across sparse layers")
    keys = (
        "final_per_head_selected_mean", "final_group_union_count_mean",
        "final_loaded_kv_tokens_mean", "pre_group_union_count_mean",
        "pre_loaded_kv_tokens_mean", "post_group_shared_count_mean",
        "post_group_union_count_mean", "post_loaded_kv_tokens_mean",
    )

    def averages(rows):
        result = {"records": len(rows), "seq_len_mean": statistics.fmean(r["seq_len"] for r in rows)}
        for key in keys:
            values = [r["gqa"][key] for r in rows if key in r["gqa"]]
            if values:
                if len(values) != len(rows):
                    raise ValueError(f"Metric {key} missing from some records")
                result[key] = statistics.fmean(values)
        return result

    return {"overall_sparse_layers": averages(sparse),
            "per_layer": {str(k): averages(v) for k, v in sorted(grouped.items())},
            "decode_steps": len(next(iter(grouped.values())))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=CONFIGS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "data/LongBench/data/qasper.jsonl")
    parser.add_argument("--sample-index", type=int, default=14, help="Zero-based Qasper row; default is closest to audited median")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.sample_index < 0 or args.max_new_tokens < 1:
        parser.error("sample index must be nonnegative and max-new-tokens positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir / f"{args.method}.profile.jsonl"
    summary_path = args.output_dir / f"{args.method}.summary.json"
    if raw.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {args.method} profile; choose a new output directory")
    os.environ.update(TWILIGHT_PROFILE="1", TWILIGHT_PROFILE_OUTPUT=str(raw.resolve()),
                      TWILIGHT_PROFILE_MODEL=str(args.model), TWILIGHT_PROFILE_CONFIG=args.method)
    import torch
    import pred
    from gas_attention.pyimpl.profiler import profiler

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run a CPU substitute")
    rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8") if line.strip()]
    if args.sample_index >= len(rows):
        raise IndexError(f"sample index {args.sample_index} exceeds dataset size {len(rows)}")
    row = rows[args.sample_index]
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    config_path = ROOT / "benchmark/configs" / CONFIGS[args.method]
    config = json.loads(config_path.read_text())
    model_name = "Meta-Llama-3.1-8B-Instruct"
    pred.seed_everything(42)
    model, tokenizer, _, budget_info, score_info = pred.load_model_and_tokenizer(
        str(args.model), model_name, torch.device("cuda"), SimpleNamespace(algo_config_path=str(config_path)))
    prompt = json.loads((ROOT / "benchmark/LongBench/config/dataset2prompt.json").read_text())["qasper"]
    max_length = json.loads((ROOT / "benchmark/LongBench/config/model2maxlen.json").read_text())[model_name]
    result = pred._predict_one(model, tokenizer, row, max_length, args.max_new_tokens, prompt,
                               "qasper", torch.device("cuda"), model_name, budget_info, score_info)
    profiler.flush()
    with raw.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    summary = summarize(records)
    summary.update({"method": args.method, "sample_index": args.sample_index,
                    "sample_sha256": hashlib.sha256(payload).hexdigest(),
                    "dataset": str(args.dataset.resolve()), "model": str(args.model.resolve()),
                    "config": config, "max_new_tokens": args.max_new_tokens,
                    "model_dtype": str(model.dtype), "prediction": result,
                    "scope": "Real single-sample LongBench question and answer decoding; sparse layers 2-31; not kernel timings",
                    "caveat": "Methods run independently; their model states and later decode tokens need not match. No Q/K/V or masks are captured."})
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"profiled {summary['decode_steps']} decode steps, 30 sparse layers; wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
