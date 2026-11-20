#!/usr/bin/env python3
"""Paired, reduced-sample Fig4a comparison against upstream LessIsMore."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

TASKS = ("niah_single_1", "niah_multikey_3", "niah_multivalue", "vt", "cwe")
TOKENS = {"niah_single_1": 128, "niah_multikey_3": 128,
          "niah_multivalue": 128, "vt": 30, "cwe": 120}
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_REFERENCE = ROOT / "benchmark/RULER/fig4a_llama31"
RELATIVE_PRED = Path("llama3-8b-chat-128k/synthetic_e1/32768/pred/gas")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reference(root, phase, task, limit):
    path = root / phase / RELATIVE_PRED / f"{task}.jsonl"
    rows = read_jsonl(path)
    if len(rows) < limit:
        raise ValueError(f"{path}: only {len(rows)} records, need {limit}")
    rows = rows[:limit]
    for row in rows:
        value = row.get("kv_metrics", {}).get("final_loaded_kv_tokens_mean")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing GAS KV metrics in {path}, index {row['index']}")
    return rows


def score(row):
    pred = str(row.get("pred", "")).strip().lower()
    refs = [str(value).lower() for value in row.get("outputs", [])]
    if not refs:
        raise ValueError(f"Missing answer in sample {row.get('index')}")
    return 100 * sum(ref in pred for ref in refs) / len(refs)


def mean(values):
    if not values:
        raise ValueError("Cannot average an empty result")
    return sum(values) / len(values)


def validate(path, refs):
    if not path.exists():
        return False
    rows = read_jsonl(path)
    if len(rows) > len(refs):
        raise ValueError(f"Too many rows in {path}; use a new output directory")
    for row, ref in zip(rows, refs):
        digest = hashlib.sha256(ref["input"].encode("utf-8")).hexdigest()
        if (row["index"] != ref["index"] or row["outputs"] != ref["outputs"]
                or row.get("input_sha256") != digest):
            raise ValueError(f"Sample mismatch in {path}; use a new output directory")
        if not math.isfinite(row["kv_metrics"]["final_loaded_kv_tokens_mean"]):
            raise ValueError(f"Invalid KV metric in {path}")
    return len(rows) == len(refs)


def worker(args):
    import torch
    from transformers import AutoTokenizer, pipeline
    from lessismore_adapter import LlamaForCausalLM, install

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run in the GPU environment")
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2", trust_remote_code=True,
    ).eval()
    recorder = install(model, args.budget)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    limit = args.cal_samples if args.phase == "calibration" else args.formal_samples
    for task in TASKS:
        refs = reference(args.reference, args.phase, task, limit)
        path = args.output / args.phase / f"lim_k{args.budget}" / f"{task}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if validate(path, refs):
            print(f"SKIP {args.phase} k={args.budget} {task}", flush=True)
            continue
        completed = len(read_jsonl(path)) if path.exists() else 0
        with path.open("a", encoding="utf-8") as handle:
            for ref in refs[completed:]:
                recorder.reset()
                prompt = ref["input"]
                with torch.inference_mode():
                    result = pipe(text_inputs=[prompt], do_sample=False, temperature=0.0,
                                  top_k=32, top_p=1.0, max_new_tokens=TOKENS[task],
                                  repetition_penalty=1.0)
                prediction = result[0][0]["generated_text"]
                if prediction.startswith(prompt):
                    prediction = prediction[len(prompt):]
                row = {"index": ref["index"], "outputs": ref["outputs"], "pred": prediction,
                       "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                       "kv_metrics": recorder.summarize()}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"DONE {args.phase} k={args.budget} {task} index={row['index']}", flush=True)


def method_stats(args, phase, budget=None):
    limit = args.cal_samples if phase == "calibration" else args.formal_samples
    task_rows = []
    for task in TASKS:
        refs = reference(args.reference, phase, task, limit)
        if budget is None:
            rows = refs
        else:
            path = args.output / phase / f"lim_k{budget}" / f"{task}.jsonl"
            if not validate(path, refs):
                raise ValueError(f"Incomplete: {path}")
            rows = read_jsonl(path)
        task_rows.append({"task": task, "samples": len(rows),
                          "score": mean([score(row) for row in rows]),
                          "loaded_kv": mean([row["kv_metrics"]["final_loaded_kv_tokens_mean"] for row in rows]),
                          "sparse_only_kv": (mean([row["kv_metrics"]["sparse_layers_only_loaded_kv_mean"] for row in rows]) if budget is not None else None)})
    return task_rows


def execute(args, phase, budget):
    print(f"RUN {phase} lim_k{budget}", flush=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--phase", phase,
           "--budget", str(budget), "--model", args.model, "--reference", str(args.reference),
           "--output", str(args.output), "--cal-samples", str(args.cal_samples),
           "--formal-samples", str(args.formal_samples)]
    subprocess.run(cmd, check=True, env=os.environ.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--cal-samples", type=int, default=3)
    parser.add_argument("--formal-samples", type=int, default=10)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", choices=["calibration", "formal"])
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    args.reference, args.output = args.reference.resolve(), args.output.resolve()
    if min(args.cal_samples, args.formal_samples, *args.budgets) <= 0:
        parser.error("sample counts and budgets must be positive")
    if args.worker:
        if args.phase is None or args.budget is None:
            parser.error("worker requires --phase and --budget")
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"model": args.model, "reference": str(args.reference), "cal_samples": args.cal_samples,
                "formal_samples": args.formal_samples, "budgets": args.budgets,
                "recent_ratio": 0.25, "selection_layers": [2, 13], "length": 32768}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Manifest differs; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    target = mean([row["loaded_kv"] for row in method_stats(args, "calibration")])
    candidates = []
    for budget in args.budgets:
        execute(args, "calibration", budget)
        rows = method_stats(args, "calibration", budget)
        candidates.append({"budget": budget, "score": mean([r["score"] for r in rows]),
                           "loaded_kv": mean([r["loaded_kv"] for r in rows])})
    within = [r for r in candidates if 1 <= r["loaded_kv"] / target <= 1.05]
    above = [r for r in candidates if r["loaded_kv"] >= target]
    selected = (max(within, key=lambda r: r["score"]) if within else
                min(above, key=lambda r: r["loaded_kv"]) if above else
                min(candidates, key=lambda r: abs(r["loaded_kv"] - target)))
    selection = {"gas_calibration_loaded_kv": target, "candidates": candidates,
                 "selected": selected, "within_5pct_above_target": bool(within),
                 "selection_rule": "best score within [1,1.05] target, otherwise nearest above, otherwise nearest"}
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"FROZEN calibration selection: {selection}", flush=True)
    execute(args, "formal", selected["budget"])
    with (args.output / "formal_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "task", "samples", "score", "loaded_kv", "sparse_only_kv"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, budget in (("gas", None), (f"lessismore_k{selected['budget']}", selected["budget"])):
            rows = method_stats(args, "formal", budget)
            for row in rows:
                writer.writerow({"method": name, **row})
            writer.writerow({"method": name, "task": "macro", "samples": sum(r["samples"] for r in rows),
                             "score": mean([r["score"] for r in rows]),
                             "loaded_kv": mean([r["loaded_kv"] for r in rows]),
                             "sparse_only_kv": mean([r["sparse_only_kv"] for r in rows]) if budget is not None else ""})
    print(f"COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
