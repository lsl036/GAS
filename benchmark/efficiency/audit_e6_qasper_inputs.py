#!/usr/bin/env python3
"""Audit exact LongBench Qasper inputs for a future sparse-kernel replay."""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LONGBENCH = ROOT / "benchmark" / "LongBench"
sys.path.insert(0, str(LONGBENCH))
from pred import build_chat  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_sample(row, index, prompt_format, tokenizer, max_length):
    # Match LongBench/pred.py: truncate before chat wrapping, then split at
    # the final question or the last 100 characters, whichever is later.
    prompt = prompt_format.format(**row)
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    original_tokens = len(ids)
    if original_tokens > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(
            ids[-half:], skip_special_tokens=True
        )
    prompt = build_chat(tokenizer, prompt, "Meta-Llama-3.1-8B-Instruct")
    split = max(len(prompt) - 100, prompt.rfind("Question:"))
    prefix, question = prompt[:split], prompt[split:]
    prefix_tokens = tokenizer(prefix, truncation=False, return_tensors="pt").input_ids.shape[-1]
    question_tokens = tokenizer(question, truncation=False, return_tensors="pt").input_ids.shape[-1] - 1
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sample_id": f"qasper:{index:06d}",
        "sample_sha256": hashlib.sha256(payload).hexdigest(),
        "dataset_length": row["length"],
        "formatted_prompt_tokens": original_tokens,
        "truncated": original_tokens > max_length,
        "prefill_tokens": prefix_tokens,
        "question_decode_tokens": question_tokens,
        "initial_generation_kv_tokens": prefix_tokens + question_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "data/LongBench/data/qasper.jsonl")
    parser.add_argument("--model", type=Path, default=Path("/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    prompts_path = LONGBENCH / "config/dataset2prompt.json"
    lengths_path = LONGBENCH / "config/model2maxlen.json"
    prompt_format = json.loads(prompts_path.read_text())["qasper"]
    max_length = json.loads(lengths_path.read_text())["Meta-Llama-3.1-8B-Instruct"]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=False)
    with args.dataset.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Qasper dataset is empty")
    results = [audit_sample(row, index, prompt_format, tokenizer, max_length) for index, row in enumerate(rows)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "qasper_input_lengths.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lengths = [row["initial_generation_kv_tokens"] for row in results]
    manifest = {
        "task": "qasper", "sample_count": len(rows), "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset), "prompt_config_sha256": file_sha256(prompts_path),
        "max_length_config_sha256": file_sha256(lengths_path), "model": str(args.model.resolve()),
        "tokenizer_class": type(tokenizer).__name__, "max_length": max_length,
        "metric": "prefill token count plus decoded question token count; excludes generated answer tokens",
        "min": min(lengths), "median": statistics.median(lengths), "max": max(lengths),
        "at_least_16384": sum(length >= 16384 for length in lengths),
        "at_least_32768": sum(length >= 32768 for length in lengths),
        "truncated_samples": sum(row["truncated"] for row in results),
    }
    (args.output_dir / "qasper_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
