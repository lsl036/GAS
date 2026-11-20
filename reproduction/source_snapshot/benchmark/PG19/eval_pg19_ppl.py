import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
BENCHMARK_DIR = ROOT / "benchmark"
DEFAULT_MODEL = "/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct"
DEFAULT_PG19_TEST = (
    "/data/home/lsl/SparseAtt/data/pg19/emozilla_pg19/data/"
    "test-00000-of-00001-29a571947c0b5ccc.parquet"
)

METHOD_CONFIGS = {
    "quest": "benchmark/configs/config_quest_8192_gqa_metrics.json",
    "quest_twi": "benchmark/configs/config_quest_twi_gqa_metrics.json",
    "ds": "benchmark/configs/config_ds_8192_llama31_gqa_metrics.json",
    "ds_twi": "benchmark/configs/config_ds_twi_llama31_gqa_metrics.json",
    "quest1024": "benchmark/configs/config_quest_1024_gqa_metrics.json",
    "quest1024_twi": "benchmark/configs/config_quest_1024_twi_gqa_metrics.json",
    "ds1024": "benchmark/configs/config_ds_1024_llama31_gqa_metrics.json",
    "ds1024_twi": "benchmark/configs/config_ds_1024_twi_llama31_gqa_metrics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pg19-test", default=DEFAULT_PG19_TEST)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--eval-len", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument(
        "--methods",
        default="full,quest,quest_twi,ds,ds_twi",
        help="Comma-separated list from full, quest, quest_twi, ds, ds_twi.",
    )
    parser.add_argument(
        "--method",
        choices=[
            "full",
            "quest",
            "quest_twi",
            "ds",
            "ds_twi",
            "quest1024",
            "quest1024_twi",
            "ds1024",
            "ds1024_twi",
        ],
    )
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--torch-extensions-dir", default="/data/tmp/torch_ext_pg19_ppl")
    parser.add_argument("--single-method", action="store_true")
    parser.add_argument(
        "--sweep-top-p",
        help="Run a top-p sweep. Format: start:stop:step, e.g. 0.0:1.0:0.1.",
    )
    parser.add_argument(
        "--sweep-base",
        choices=["quest1024", "ds1024"],
        default="ds1024",
        help="Base top-k selector used for generated top-p configs.",
    )
    parser.add_argument(
        "--sweep-base-config",
        help="Optional base selector config path. Overrides the built-in config for --sweep-base.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write generated configs and commands; do not run model inference.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=28.0,
        help="Skip running unless GPU free memory is at least this many GiB.",
    )
    parser.add_argument(
        "--ignore-gpu-memory-check",
        action="store_true",
        help="Run even if the free-memory check fails.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stats(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {}
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "p50": vals[int((len(vals) - 1) * 0.50)],
        "p90": vals[int((len(vals) - 1) * 0.90)],
        "max": vals[-1],
    }


def summarize_profile(profile_path: Path) -> Dict[str, Dict[str, float]]:
    values: Dict[str, List[float]] = {}
    for record in read_jsonl(profile_path):
        for key, value in record.get("gqa", {}).items():
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(float(value))
        for key in ("seq_len", "avg_selected", "max_selected"):
            if key in record and isinstance(record[key], (int, float)):
                values.setdefault(key, []).append(float(record[key]))
    return {key: stats(vals) for key, vals in sorted(values.items())}


def load_eval_sequences(args, tokenizer) -> List[torch.Tensor]:
    dataset = load_dataset(
        "parquet",
        data_files={"test": args.pg19_test},
        split="test",
    )
    required_len = args.seq_len + 1
    stride = args.stride or args.seq_len
    sequences = []
    for row in dataset:
        text = row[args.text_field]
        ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, max(0, len(ids) - required_len + 1), stride):
            chunk = ids[start : start + required_len]
            if len(chunk) == required_len:
                sequences.append(torch.tensor(chunk, dtype=torch.long))
                if len(sequences) >= args.num_samples:
                    return sequences
    if len(sequences) < args.num_samples:
        raise ValueError(
            f"Collected only {len(sequences)} sequences with length {required_len}; "
            f"requested {args.num_samples}."
        )
    return sequences


def enable_sparse_if_needed(model, config_path: Optional[str]) -> None:
    if not config_path:
        return
    sys.path.insert(0, str(ROOT))
    from twilight.pyimpl import enable_sparse_attention

    with open(config_path, "r", encoding="utf-8") as f:
        sparse_config = json.load(f)
    enable_sparse_attention(
        model,
        sparse_config=sparse_config,
        enable_budget_info=False,
        enable_score_info=False,
    )


@torch.no_grad()
def evaluate_single_method(args) -> dict:
    output_dir = Path(args.output_dir)
    method_dir = output_dir / args.method
    method_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    group_size = num_heads // num_kv_heads

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
        model_max_length=sys.maxsize,
    )
    sequences = load_eval_sequences(args, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    ).eval()
    enable_sparse_if_needed(model, args.config)

    total_nll = 0.0
    total_tokens = 0
    sample_records = []
    prefill_len = args.seq_len - args.eval_len
    device = next(model.parameters()).device

    for sample_idx, ids in enumerate(tqdm(sequences, desc=f"PG-19 {args.method}")):
        input_ids = ids[:prefill_len].unsqueeze(0).to(device)
        eval_ids = ids[prefill_len : args.seq_len + 1].to(device)
        output = model(input_ids=input_ids, use_cache=True)
        past_key_values = output.past_key_values

        sample_nll = 0.0
        for step in range(args.eval_len):
            cur_id = eval_ids[step].view(1, 1)
            target_id = eval_ids[step + 1].view(1)
            output = model(
                input_ids=cur_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, :].float()
            nll = torch.nn.functional.cross_entropy(logits, target_id, reduction="sum")
            sample_nll += float(nll.item())

        sample_ppl = math.exp(sample_nll / args.eval_len)
        total_nll += sample_nll
        total_tokens += args.eval_len
        sample_records.append(
            {
                "sample_idx": sample_idx,
                "nll": sample_nll,
                "tokens": args.eval_len,
                "ppl": sample_ppl,
            }
        )

    profile_output = os.environ.get("TWILIGHT_PROFILE_OUTPUT", "")
    profile_path = Path(profile_output) if profile_output else None
    gqa_summary = summarize_profile(profile_path) if profile_path else {}
    mean_decode_seq_len = prefill_len + (args.eval_len + 1) / 2

    if args.method == "full":
        gqa_summary = {
            "group_size": {"mean": float(group_size)},
            "per_head_selected_mean": {"mean": float(mean_decode_seq_len)},
            "group_union_count_mean": {"mean": float(mean_decode_seq_len)},
            "loaded_kv_tokens_mean": {"mean": float(mean_decode_seq_len * num_kv_heads)},
            "union_amplification_mean": {"mean": 1.0},
        }

    result = {
        "method": args.method,
        "config": args.config,
        "model": args.model,
        "pg19_test": args.pg19_test,
        "seq_len": args.seq_len,
        "eval_len": args.eval_len,
        "num_samples": len(sequences),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "group_size": group_size,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "ppl": math.exp(total_nll / total_tokens),
        "mean_sample_ppl": statistics.fmean(row["ppl"] for row in sample_records),
        "gqa": gqa_summary,
        "profile_path": str(profile_path) if profile_path else "",
    }

    with open(method_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for row in sample_records:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with open(method_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


def metric_mean(result: dict, key: str) -> float:
    return float(result.get("gqa", {}).get(key, {}).get("mean", 0.0))


def write_summary(output_dir: Path, results: List[dict], command_lines: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pg19_ppl_results.jsonl", "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "# PG-19 PPL Motivation Results",
        "",
        "## Commands",
        "",
        "```bash",
        *command_lines,
        "```",
        "",
        "## Summary",
        "",
        "| method | ppl | tokens | per-head selected | group union | loaded KV tokens | union amp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {method} | {ppl:.4f} | {tokens} | {per_head:.2f} | {union:.2f} | "
            "{loaded:.2f} | {amp:.2f} |".format(
                method=row["method"],
                ppl=row["ppl"],
                tokens=row["total_tokens"],
                per_head=metric_mean(row, "per_head_selected_mean"),
                union=metric_mean(row, "group_union_count_mean"),
                loaded=metric_mean(row, "loaded_kv_tokens_mean"),
                amp=metric_mean(row, "union_amplification_mean"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `per-head selected` is the apparent sparse budget per query head.",
            "- `group union` is the number of distinct tokens loaded for one KV head after GQA union.",
            "- `loaded KV tokens` is `group union` summed across KV heads.",
            "- `union amp > 1` indicates extra KV-cache load caused by per-head sparse masks under GQA.",
        ]
    )
    (output_dir / "pg19_ppl_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "run_pg19_ppl_commands.md").write_text(
        "# Commands\n\n```bash\n" + "\n".join(command_lines) + "\n```\n",
        encoding="utf-8",
    )


def parse_sweep_values(spec: str) -> List[float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError("--sweep-top-p must have format start:stop:step")
    start, stop, step = (float(part) for part in parts)
    if step <= 0:
        raise ValueError("Sweep step must be positive.")
    values = []
    cur = start
    eps = step / 1000
    while cur <= stop + eps:
        values.append(round(cur, 10))
        cur += step
    return values


def top_p_label(value: float) -> str:
    return f"p{int(round(value * 100)):03d}"


def build_top_p_config(base_config_path: Path, threshold: float) -> dict:
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["weight_estimator"] = {
        "type": "min_max_quant",
        "quant_bit": 4,
    }
    config["weight_pruner"] = {
        "type": "top_p",
        "threshold": threshold,
    }
    config["gqa_refinement"] = {
        "metrics_only": True,
        "mode": "paper_union_metric",
    }
    return config


def get_gpu_free_gb() -> Optional[float]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return None
    values = [float(line.strip()) for line in output.splitlines() if line.strip()]
    if not values:
        return None
    return max(values) / 1024


def command_to_string(command: List[str]) -> str:
    return " ".join(command)


def write_sweep_summary(
    output_dir: Path,
    results: List[dict],
    command_lines: List[str],
    base_method: str,
    sweep_values: List[float],
    ran: bool,
    memory_message: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sweep_results.jsonl", "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "# PG-19 Top-p Sweep",
        "",
        f"- base selector: `{base_method}`",
        f"- top-p values: `{', '.join(str(v) for v in sweep_values)}`",
        f"- status: `{'ran' if ran else 'prepared_only'}`",
        f"- memory check: {memory_message}",
        "",
        "## Commands",
        "",
        "```bash",
        *command_lines,
        "```",
        "",
        "## Sweep Results",
        "",
        "| p | method | ppl | tokens | per-head selected | group union | loaded KV tokens | union amp |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {p:.1f} | {method} | {ppl:.4f} | {tokens} | {per_head:.2f} | "
            "{union:.2f} | {loaded:.2f} | {amp:.2f} |".format(
                p=float(row["top_p"]),
                method=row["method"],
                ppl=row["ppl"],
                tokens=row["total_tokens"],
                per_head=metric_mean(row, "per_head_selected_mean"),
                union=metric_mean(row, "group_union_count_mean"),
                loaded=metric_mean(row, "loaded_kv_tokens_mean"),
                amp=metric_mean(row, "union_amplification_mean"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- `p=0.0` is a boundary/debug point; use `0.1..1.0` for formal interpretation.",
            "- If PPL changes little while `loaded KV tokens` grows, the added top-p budget is low-value.",
            "- If `union amp` stays well above 1, per-head sparse masks are still amplified by GQA union.",
        ]
    )
    (output_dir / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "run_top_p_sweep_commands.md").write_text(
        "# Commands\n\n```bash\n" + "\n".join(command_lines) + "\n```\n",
        encoding="utf-8",
    )


def run_top_p_sweep(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    sweep_values = parse_sweep_values(args.sweep_top_p)
    base_config = Path(args.sweep_base_config) if args.sweep_base_config else ROOT / METHOD_CONFIGS[args.sweep_base]
    commands = []
    results = []

    for value in sweep_values:
        label = top_p_label(value)
        method = f"{args.sweep_base}_twi_{label}"
        config_path = config_dir / f"{method}.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(build_top_p_config(base_config, value), f, indent=2, sort_keys=True)

        torch_ext_dir = Path(args.torch_extensions_dir) / method
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-method",
            "--method",
            args.sweep_base,
            "--model",
            args.model,
            "--pg19-test",
            args.pg19_test,
            "--seq-len",
            str(args.seq_len),
            "--eval-len",
            str(args.eval_len),
            "--num-samples",
            str(args.num_samples),
            "--output-dir",
            str(output_dir / "runs" / method),
            "--torch-extensions-dir",
            str(torch_ext_dir),
            "--config",
            str(config_path),
        ]
        commands.append(command_to_string(command))

    free_gb = get_gpu_free_gb()
    if free_gb is None:
        memory_message = "could not query GPU memory"
        memory_ok = False
    else:
        memory_message = f"{free_gb:.2f} GiB free; threshold {args.min_free_gb:.2f} GiB"
        memory_ok = free_gb >= args.min_free_gb

    should_run = not args.prepare_only and (memory_ok or args.ignore_gpu_memory_check)
    if not should_run:
        write_sweep_summary(
            output_dir,
            results,
            commands,
            args.sweep_base,
            sweep_values,
            ran=False,
            memory_message=memory_message,
        )
        print(f"prepared sweep: {output_dir / 'sweep_summary.md'}")
        print(f"memory check: {memory_message}")
        return

    for command_line, value in zip(commands, sweep_values):
        command = command_line.split(" ")
        method = f"{args.sweep_base}_twi_{top_p_label(value)}"
        env = os.environ.copy()
        env["HF_HOME"] = env.get("HF_HOME", "/data/tmp/hf_home")
        env["HF_DATASETS_CACHE"] = env.get("HF_DATASETS_CACHE", "/data/tmp/hf_datasets_cache")
        env["TORCH_EXTENSIONS_DIR"] = str(Path(args.torch_extensions_dir) / method)
        env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", "8.0")
        env["TWILIGHT_PROFILE"] = "1"
        env["TWILIGHT_PROFILE_OUTPUT"] = str(output_dir / "runs" / method / args.sweep_base / "profile.jsonl")
        env["TWILIGHT_PROFILE_MODEL"] = args.model
        env["TWILIGHT_PROFILE_CONFIG"] = str(output_dir / "generated_configs" / f"{method}.json")
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)
        result_path = output_dir / "runs" / method / args.sweep_base / "result.json"
        result = json.loads(result_path.read_text())
        result["top_p"] = value
        result["method"] = method
        results.append(result)
        write_sweep_summary(
            output_dir,
            results,
            commands,
            args.sweep_base,
            sweep_values,
            ran=True,
            memory_message=memory_message,
        )

    print(f"sweep summary: {output_dir / 'sweep_summary.md'}")


def run_multi_method(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    results = []
    commands = []

    for method in methods:
        if method != "full" and method not in METHOD_CONFIGS:
            raise ValueError(f"Unsupported method: {method}")
        config = args.config if method == args.method and args.config else METHOD_CONFIGS.get(method)
        profile_path = output_dir / method / "profile.jsonl"
        torch_ext_dir = Path(args.torch_extensions_dir) / method
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-method",
            "--method",
            method,
            "--model",
            args.model,
            "--pg19-test",
            args.pg19_test,
            "--seq-len",
            str(args.seq_len),
            "--eval-len",
            str(args.eval_len),
            "--num-samples",
            str(args.num_samples),
            "--output-dir",
            str(output_dir),
            "--torch-extensions-dir",
            str(torch_ext_dir),
        ]
        if config:
            command.extend(["--config", str(ROOT / config)])

        env = os.environ.copy()
        env["HF_HOME"] = env.get("HF_HOME", "/data/tmp/hf_home")
        env["HF_DATASETS_CACHE"] = env.get("HF_DATASETS_CACHE", "/data/tmp/hf_datasets_cache")
        env["TORCH_EXTENSIONS_DIR"] = str(torch_ext_dir)
        env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", "8.0")
        if method != "full":
            env["TWILIGHT_PROFILE"] = "1"
            env["TWILIGHT_PROFILE_OUTPUT"] = str(profile_path)
            env["TWILIGHT_PROFILE_MODEL"] = args.model
            env["TWILIGHT_PROFILE_CONFIG"] = config or ""
        else:
            env.pop("TWILIGHT_PROFILE", None)
            env.pop("TWILIGHT_PROFILE_OUTPUT", None)

        commands.append(" ".join(command))
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)
        results.append(json.loads((output_dir / method / "result.json").read_text()))
        write_summary(output_dir, results, commands)

    print(f"summary: {output_dir / 'pg19_ppl_summary.md'}")


def main() -> None:
    args = parse_args()
    if args.sweep_top_p:
        run_top_p_sweep(args)
    elif args.single_method:
        evaluate_single_method(args)
    else:
        run_multi_method(args)


if __name__ == "__main__":
    main()
