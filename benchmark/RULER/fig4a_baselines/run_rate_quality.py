#!/usr/bin/env python3
"""Paired LessIsMore 4%/8% quality and logical loaded-KV measurements."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import run_quality_matrix as quality

ROOT = quality.ROOT
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "benchmark/results_quality/lessismore_rate_13task_10"
RATES = {"4pct": 0.04, "8pct": 0.08}


def output_file(root, suite, rate, length, task):
    path = root / suite / rate
    if suite == "ruler":
        path /= str(length)
    return path / f"{task}.jsonl"


def check_existing(path, suite, task, source):
    rows = quality.validate_checkpoint(path, suite, task, source)
    for row in rows:
        if "kv_metrics" not in row:
            raise ValueError(f"Missing logical KV metrics in {path}; use a new output directory")
    return rows


def setup(args):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, pipeline
    from lessismore_adapter import LlamaForCausalLM
    from lessismore_rate_adapter import install_rate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    seed = 43 if args.suite == "ruler" else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="flash_attention_2", trust_remote_code=True).eval()
    if any(str(device) in ("cpu", "disk") for device in model.hf_device_map.values()):
        raise RuntimeError("Model was offloaded to CPU/disk; refusing a different inference protocol")
    tracker = install_rate(model, RATES[args.rate])
    if args.suite == "ruler" and tokenizer.pad_token is None:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, tracker, pipeline


def run_ruler(args, model, tokenizer, tracker, pipeline):
    import torch
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    incomplete = False
    for length in args.lengths:
        for task in args.ruler_tasks:
            source = quality.source_rows("ruler", length, task, args.samples)
            path = output_file(args.output, "ruler", args.rate, length, task)
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = check_existing(path, "ruler", task, source)
            print(f"RUN ruler rate={args.rate} length={length} task={task} {len(existing)}/{args.samples}", flush=True)
            max_tokens = 128 if task.startswith("niah") else {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32, "qa_2": 32}[task]
            with path.open("a", encoding="utf-8") as handle:
                for index in range(len(existing), len(source)):
                    item = source[index]
                    tracker.reset()
                    try:
                        with torch.inference_mode():
                            generated = pipe(text_inputs=[item["input"]], do_sample=False,
                                temperature=0.0, top_k=32, top_p=1.0,
                                max_new_tokens=max_tokens, repetition_penalty=1.0)
                        pred = generated[0][0]["generated_text"]
                        if pred.startswith(item["input"]):
                            pred = pred[len(item["input"]):]
                    except torch.cuda.OutOfMemoryError as exc:
                        print(f"OOM ruler rate={args.rate} length={length} task={task} sample={index}: {exc}", flush=True)
                        incomplete = True
                        break
                    record = {"sample_index": index, "index": item["index"],
                              "source_sha256": quality.digest(item), "pred": pred,
                              "outputs": item["outputs"], "kv_metrics": tracker.snapshot()}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"DONE ruler rate={args.rate} length={length} task={task} {index+1}/{args.samples} "
                          f"kv={record['kv_metrics']}", flush=True)
            torch.cuda.empty_cache()
    return not incomplete


def run_longbench(args, model, tokenizer, tracker):
    import torch
    sys.path.insert(0, str(quality.LONG_CODE))
    import pred

    config = quality.LONG_CODE / "config"
    prompts = json.loads((config / "dataset2prompt.json").read_text())
    max_gen = json.loads((config / "dataset2maxlen.json").read_text())
    max_length = json.loads((config / "model2maxlen.json").read_text())["Meta-Llama-3.1-8B-Instruct"]
    incomplete = False
    for task in args.long_tasks:
        key = "repobench-p" if task == "repobench" else task
        source = quality.source_rows("longbench", None, task, args.samples)
        quality.check_longbench_manifest(task, source)
        path = output_file(args.output, "longbench", args.rate, None, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = check_existing(path, "longbench", task, source)
        print(f"RUN longbench rate={args.rate} task={task} {len(existing)}/{args.samples}", flush=True)
        with path.open("a", encoding="utf-8") as handle:
            for index in range(len(existing), len(source)):
                item = source[index]
                tracker.reset()
                tracker.last_summary = None
                try:
                    record = pred._predict_one(model, tokenizer, item, max_length, max_gen[key],
                        prompts[key], key, torch.device("cuda"), "Meta-Llama-3.1-8B-Instruct",
                        tracker, quality.NoMetrics())
                except torch.cuda.OutOfMemoryError as exc:
                    print(f"OOM longbench rate={args.rate} task={task} sample={index}: {exc}", flush=True)
                    incomplete = True
                    break
                record.update({"_e5_sample_id": f"{task}:{index:06d}",
                               "_e5_sample_hash": quality.digest(item), "_e5_row_index": index,
                               "kv_metrics": tracker.last_summary})
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"DONE longbench rate={args.rate} task={task} {index+1}/{args.samples} "
                      f"kv={record['kv_metrics']}", flush=True)
        torch.cuda.empty_cache()
    return not incomplete


def summarize(args):
    sys.path.insert(0, str(quality.LONG_CODE))
    from summarize_e5_longbench import metric_modules, score_record
    metrics = metric_modules()
    output = []
    for suite in ("ruler", "longbench"):
        for rate in RATES:
            for length in args.lengths if suite == "ruler" else (None,):
                tasks = args.ruler_tasks if suite == "ruler" else args.long_tasks
                scores, loaded, sparse, budgets = [], [], [], []
                total = 0
                for task in tasks:
                    source = quality.source_rows(suite, length, task, args.samples)
                    if suite == "longbench":
                        quality.check_longbench_manifest(task, source)
                    path = output_file(args.output, suite, rate, length, task)
                    records = check_existing(path, suite, task, source)
                    total += len(records)
                    task_score = task_loaded = task_sparse = task_min = task_max = ""
                    if len(records) == args.samples:
                        if suite == "ruler":
                            task_score = quality.ruler_score(task, records)
                        else:
                            sample_scores = [score_record(task, row, metrics[task][1]) for row in records]
                            if any(failed for _, failed in sample_scores):
                                raise ValueError(f"Invalid predictions: {path}")
                            task_score = 100 * sum(score for score, _ in sample_scores) / len(sample_scores)
                        scores.append(task_score)
                        measured = [row["kv_metrics"] for row in records]
                        if all(isinstance(value, dict) for value in measured):
                            task_loaded = sum(value["final_loaded_kv_tokens_mean"] for value in measured) / len(measured)
                            task_sparse = sum(value["sparse_layers_only_loaded_kv_mean"] for value in measured) / len(measured)
                            task_min = min(value["budget_min"] for value in measured)
                            task_max = max(value["budget_max"] for value in measured)
                            loaded.append(task_loaded)
                            sparse.append(task_sparse)
                            budgets.append((task_min, task_max))
                    output.append([suite, rate, length or "", task, len(records), args.samples,
                                   task_score, task_loaded, task_sparse, task_min, task_max])
                complete = len(scores) == len(tasks)
                kv_complete = len(loaded) == len(tasks)
                output.append([suite, rate, length or "", "macro", total, args.samples * len(tasks),
                               sum(scores) / len(tasks) if complete else "",
                               sum(loaded) / len(tasks) if kv_complete else "",
                               sum(sparse) / len(tasks) if kv_complete else "",
                               min(value[0] for value in budgets) if kv_complete else "",
                               max(value[1] for value in budgets) if kv_complete else ""])
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "task_scores_and_kv.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", "rate", "context_length", "task", "n_predictions", "n_expected",
                         "score", "mean_loaded_kv", "sparse_only_kv", "budget_min", "budget_max"])
        writer.writerows(output)
    temporary.replace(path)
    print(f"WROTE {path}; incomplete macro scores or KV metrics are blank", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("preflight", "ruler", "longbench", "summarize"), required=True)
    parser.add_argument("--rate", choices=RATES)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--model", default=quality.MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(quality.LENGTHS))
    parser.add_argument("--ruler-tasks", nargs="+", default=list(quality.RULER_TASKS), choices=quality.RULER_TASKS)
    parser.add_argument("--long-tasks", nargs="+", default=list(quality.LONG_TASKS), choices=quality.LONG_TASKS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.samples <= 0 or any(length not in quality.LENGTHS for length in args.lengths):
        parser.error("invalid sample count or RULER length")
    if args.suite == "preflight":
        quality.verify_sources(args)
        return
    if args.suite == "summarize":
        summarize(args)
        return
    if args.rate is None:
        parser.error("--rate is required for inference")
    protocol = {"model": args.model, "samples": args.samples, "lengths": args.lengths,
                "ruler_tasks": args.ruler_tasks, "long_tasks": args.long_tasks,
                "dtype": "bfloat16", "recent_ratio": 0.25, "selection_layers": [2, 13],
                "rates": RATES, "budget_rule": "max(1, floor(current KV length * rate)) per decode step",
                "kv_metric": "logical final attention mask, layers 2-31 including full selection layers"}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol differs from existing outputs; use a new output directory")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    model, tokenizer, tracker, pipeline = setup(args)
    complete = (run_ruler(args, model, tokenizer, tracker, pipeline) if args.suite == "ruler"
                else run_longbench(args, model, tokenizer, tracker))
    if not complete:
        raise SystemExit("Incomplete due to OOM; rerun after resolving GPU memory pressure")
    print(f"COMPLETE suite={args.suite} rate={args.rate}", flush=True)


if __name__ == "__main__":
    main()
