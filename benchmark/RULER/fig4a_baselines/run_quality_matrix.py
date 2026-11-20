#!/usr/bin/env python3
"""Run LessIsMore on paired RULER/LongBench prefixes with resumable outputs."""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RULER = ROOT / "benchmark/RULER/scripts/dataset/llama3-8b-chat-128k/synthetic"
LONG_DIR = Path("/data/home/lsl/SparseAtt/data/LongBench/data")
LONG_CODE = ROOT / "benchmark/LongBench"
MANIFEST = ROOT / "benchmark/results_longbench/e5_longbench_completion/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "benchmark/results_quality/lessismore_13task_10"
MODEL = "/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct"
RULER_TASKS = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
               "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
               "vt", "cwe", "fwe", "qa_1", "qa_2")
LONG_TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
              "gov_report", "qmsum", "multi_news", "triviaqa", "passage_retrieval_en", "lcc", "repobench")
LENGTHS = (16384, 32768, 65536)


def read_jsonl(path, limit=None):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) == limit:
                    break
    return rows


def digest(record):
    data = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def source_rows(suite, length, task, count):
    if suite == "ruler":
        path = RULER / str(length) / "data" / task / "validation.jsonl"
    else:
        key = "repobench-p" if task == "repobench" else task
        path = LONG_DIR / f"{key}.jsonl"
    rows = read_jsonl(path, count)
    if len(rows) != count:
        raise ValueError(f"{path}: found {len(rows)} records, expected {count}")
    return rows


def check_longbench_manifest(task, rows):
    entries = [row for row in read_jsonl(MANIFEST) if row["task"] == task][:len(rows)]
    if len(entries) != len(rows):
        raise ValueError(f"E5 manifest has too few records for {task}")
    for index, (sample, entry) in enumerate(zip(rows, entries)):
        if entry["sample_id"] != f"{task}:{index:06d}" or entry["sample_hash"] != digest(sample):
            raise ValueError(f"LongBench sample mismatch against E5: {task}/{index}")


def output_file(root, suite, budget, length, task):
    location = root / suite / f"k{budget}"
    if suite == "ruler":
        location /= str(length)
    return location / f"{task}.jsonl"


def validate_checkpoint(path, suite, task, source):
    rows = read_jsonl(path) if path.exists() else []
    if len(rows) > len(source):
        raise ValueError(f"Too many predictions in {path}")
    for index, (record, item) in enumerate(zip(rows, source)):
        if suite == "ruler":
            valid = (record.get("sample_index") == index and record.get("source_sha256") == digest(item)
                     and record.get("outputs") == item["outputs"])
        else:
            valid = (record.get("_e5_sample_id") == f"{task}:{index:06d}"
                     and record.get("_e5_sample_hash") == digest(item)
                     and record.get("answers") == item["answers"]
                     and record.get("all_classes") == item["all_classes"])
        if not valid or not isinstance(record.get("pred"), str):
            raise ValueError(f"Invalid or unpaired prediction: {path}, row {index}")
    return rows


def verify_sources(args):
    for length in args.lengths:
        for task in args.ruler_tasks:
            source_rows("ruler", length, task, args.samples)
    for task in args.long_tasks:
        rows = source_rows("longbench", None, task, args.samples)
        check_longbench_manifest(task, rows)
    if not Path(args.model).is_dir():
        raise FileNotFoundError(args.model)
    print(f"PREFLIGHT OK: RULER {len(args.lengths)} lengths x {len(args.ruler_tasks)} tasks; "
          f"LongBench {len(args.long_tasks)} tasks; {args.samples} samples/task", flush=True)


def setup_model(args):
    import torch
    import numpy as np
    from transformers import AutoTokenizer, pipeline
    from lessismore_adapter import LlamaForCausalLM, install

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    random.seed(43 if args.suite == "ruler" else 42)
    np.random.seed(43 if args.suite == "ruler" else 42)
    torch.manual_seed(43 if args.suite == "ruler" else 42)
    torch.cuda.manual_seed_all(43 if args.suite == "ruler" else 42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="flash_attention_2", trust_remote_code=True).eval()
    if any(str(device) in ("cpu", "disk") for device in model.hf_device_map.values()):
        raise RuntimeError("Model was offloaded to CPU/disk; refusing a different inference protocol")
    install(model, args.budget, track_metrics=False)
    if args.suite == "ruler" and tokenizer.pad_token is None:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, pipeline


def ruler_generate(args, model, tokenizer, pipeline):
    import torch
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    incomplete = False
    for length in args.lengths:
        for task in args.ruler_tasks:
            source = source_rows("ruler", length, task, args.samples)
            path = output_file(args.output, "ruler", args.budget, length, task)
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = validate_checkpoint(path, "ruler", task, source)
            print(f"RUN ruler k={args.budget} length={length} task={task} {len(existing)}/{args.samples}", flush=True)
            max_tokens = 128 if task.startswith("niah") else {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32, "qa_2": 32}[task]
            with path.open("a", encoding="utf-8") as handle:
                for index in range(len(existing), len(source)):
                    item = source[index]
                    try:
                        with torch.inference_mode():
                            generated = pipe(text_inputs=[item["input"]], do_sample=False,
                                temperature=0.0, top_k=32, top_p=1.0,
                                max_new_tokens=max_tokens, repetition_penalty=1.0)
                        pred = generated[0][0]["generated_text"]
                        if pred.startswith(item["input"]):
                            pred = pred[len(item["input"]):]
                    except torch.cuda.OutOfMemoryError as exc:
                        print(f"OOM ruler k={args.budget} length={length} task={task} sample={index}: {exc}", flush=True)
                        incomplete = True
                        break
                    record = {"sample_index": index, "index": item["index"], "source_sha256": digest(item),
                              "pred": pred, "outputs": item["outputs"]}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"DONE ruler k={args.budget} length={length} task={task} {index+1}/{args.samples}", flush=True)
            torch.cuda.empty_cache()
    return not incomplete


class NoMetrics:
    def get_total_avg_budget(self):
        return None

    def get_total_avg_score(self):
        return None

    def reset(self):
        pass


def longbench_generate(args, model, tokenizer):
    import torch
    sys.path.insert(0, str(LONG_CODE))
    import pred

    config = LONG_CODE / "config"
    prompts = json.loads((config / "dataset2prompt.json").read_text())
    max_gen = json.loads((config / "dataset2maxlen.json").read_text())
    max_length = json.loads((config / "model2maxlen.json").read_text())["Meta-Llama-3.1-8B-Instruct"]
    incomplete = False
    for task in args.long_tasks:
        key = "repobench-p" if task == "repobench" else task
        source = source_rows("longbench", None, task, args.samples)
        check_longbench_manifest(task, source)
        path = output_file(args.output, "longbench", args.budget, None, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = validate_checkpoint(path, "longbench", task, source)
        print(f"RUN longbench k={args.budget} task={task} {len(existing)}/{args.samples}", flush=True)
        try:
            pred.get_pred(model, tokenizer, source, max_length, max_gen[key], prompts[key], key,
                          torch.device("cuda"), "Meta-Llama-3.1-8B-Instruct", NoMetrics(), NoMetrics(),
                          checkpoint_path=str(path), checkpoint_task_name=task)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"OOM longbench k={args.budget} task={task}: {exc}", flush=True)
            incomplete = True
        torch.cuda.empty_cache()
    return not incomplete


def ruler_score(task, records):
    values = []
    for record in records:
        pred = re.sub(r"[\x00-\x1f]", "\n", record["pred"].strip()).strip().lower()
        refs = [str(value).lower() for value in record["outputs"]]
        if not refs:
            raise ValueError(f"Empty references for {task}")
        hits = [float(value in pred) for value in refs]
        values.append(max(hits) if task.startswith("qa_") else sum(hits) / len(hits))
    return round(100 * sum(values) / len(values), 2)


def summarize(args):
    sys.path.insert(0, str(LONG_CODE))
    from summarize_e5_longbench import metric_modules, score_record
    metrics = metric_modules()
    rows = []
    for suite in ("ruler", "longbench"):
        for budget in (128, 512):
            for length in args.lengths if suite == "ruler" else (None,):
                tasks = args.ruler_tasks if suite == "ruler" else args.long_tasks
                task_scores = []
                total = 0
                for task in tasks:
                    source = source_rows(suite, length, task, args.samples)
                    if suite == "longbench":
                        check_longbench_manifest(task, source)
                    path = output_file(args.output, suite, budget, length, task)
                    records = validate_checkpoint(path, suite, task, source)
                    total += len(records)
                    value = ""
                    if len(records) == args.samples:
                        if suite == "ruler":
                            value = ruler_score(task, records)
                        else:
                            scores = [score_record(task, record, metrics[task][1]) for record in records]
                            if any(failed for _, failed in scores):
                                raise ValueError(f"Invalid LongBench predictions: {path}")
                            value = 100 * sum(score for score, _ in scores) / len(scores)
                        task_scores.append(value)
                    rows.append([suite, budget, length or "", task, len(records), args.samples, value])
                macro = sum(task_scores) / len(tasks) if len(task_scores) == len(tasks) else ""
                rows.append([suite, budget, length or "", "macro", total, args.samples * len(tasks), macro])
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "task_scores.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", "budget", "context_length", "task", "n_predictions", "n_expected", "score"])
        writer.writerows(rows)
    temporary.replace(destination)
    print(f"WROTE {destination}; macro scores are blank until all tasks are complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("preflight", "ruler", "longbench", "summarize"), required=True)
    parser.add_argument("--budget", type=int, choices=(128, 512))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--ruler-tasks", nargs="+", default=list(RULER_TASKS), choices=RULER_TASKS)
    parser.add_argument("--long-tasks", nargs="+", default=list(LONG_TASKS), choices=LONG_TASKS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.samples <= 0 or any(length not in LENGTHS for length in args.lengths):
        parser.error("invalid sample count or RULER length")
    if args.suite == "preflight":
        verify_sources(args)
        return
    if args.suite == "summarize":
        summarize(args)
        return
    if args.budget is None:
        parser.error("--budget is required for inference")
    protocol = {"model": args.model, "samples": args.samples, "lengths": args.lengths,
                "ruler_tasks": args.ruler_tasks, "long_tasks": args.long_tasks,
                "dtype": "bfloat16", "recent_ratio": 0.25, "selection_layers": [2, 13]}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol differs from existing outputs; use a new output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    model, tokenizer, pipeline = setup_model(args)
    complete = (ruler_generate(args, model, tokenizer, pipeline) if args.suite == "ruler"
                else longbench_generate(args, model, tokenizer))
    if not complete:
        raise SystemExit("Incomplete due to OOM; rerun after resolving GPU memory pressure")
    print(f"COMPLETE suite={args.suite} k={args.budget}", flush=True)


if __name__ == "__main__":
    main()
