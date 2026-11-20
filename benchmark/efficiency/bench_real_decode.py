"""Measure real model decode steps, including the selected attention implementation.

Run one method per process so the Twilight monkey patch and CUDA allocations do
not leak between methods. Inputs can be one RULER JSONL sample.
"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONFIGS = {
    "patched_full": "config_full.json",
    "quest": "config_quest_8192.json",
    "quest_twi": "config_quest_twi.json",
    "gas": "config_quest_twi_gqa_group_max_p080_exception16.json",
    "gas_fused": "config_quest_twi_gqa_group_max_p080_exception16_cuda_fused.json",
}


def prompt_ids(tokenizer, length, batch, device):
    text = (
        "The grass is green. The sky is blue. The sun is yellow. "
        "A long document contains information to be read carefully. "
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise ValueError("Tokenizer returned no prompt tokens")
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    repeat = (length - len(bos) + len(tokens) - 1) // len(tokens)
    ids = (bos + tokens * repeat)[:length]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).repeat(batch, 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True, choices=["native_fa2", *CONFIGS])
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--prompt-jsonl", type=Path, help="RULER JSONL file with an input field")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based line in the JSONL file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--feeding", choices=["greedy", "teacher"], default="greedy")
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--stop-on-eos", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.context_length, args.batch_size, args.steps, args.rounds) < 1 or args.warmup_steps < 0:
        parser.error("length, batch, steps and rounds must be positive; warmup must be nonnegative")
    if args.method == "gas_fused" and args.dtype != "fp16":
        parser.error("the current GAS CUDA fused kernel requires FP16")
    if args.save_logits and args.feeding != "teacher":
        parser.error("--save-logits requires --feeding teacher for matched histories")
    if args.stop_on_eos and args.feeding != "greedy":
        parser.error("--stop-on-eos requires --feeding greedy")
    if args.sample_index < 0:
        parser.error("sample-index must be nonnegative")
    if os.environ.get("TWILIGHT_PROFILE", "0") != "0":
        parser.error("Set TWILIGHT_PROFILE=0; stage profiling changes timing")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1234)
    device = torch.device("cuda:0")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="flash_attention_2",
        local_files_only=True, low_cpu_mem_usage=True,
    ).to(device).eval()
    config_path = None
    config = None
    if args.method != "native_fa2":
        from gas_attention.pyimpl import enable_sparse_attention

        config_path = ROOT / "benchmark" / "configs" / CONFIGS[args.method]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        enable_sparse_attention(model, config, enable_budget_info=False, enable_score_info=False)

    sample = None
    if args.prompt_jsonl:
        with args.prompt_jsonl.open(encoding="utf-8") as stream:
            sample = next((json.loads(line) for index, line in enumerate(stream)
                           if index == args.sample_index), None)
        if sample is None or not isinstance(sample.get("input"), str):
            raise ValueError("JSONL sample is missing or has no string input")
        encoded = tokenizer(sample["input"], return_tensors="pt", add_special_tokens=False)
        prompt = encoded.input_ids.to(device).repeat(args.batch_size, 1)
    else:
        prompt = prompt_ids(tokenizer, args.context_length, args.batch_size, device)
    context_length = prompt.shape[1]
    prompt_digest = hashlib.sha256(prompt[0].cpu().numpy().tobytes()).hexdigest()
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = eos_ids or []

    def is_eos(ids):
        return bool(args.stop_on_eos and eos_ids and
                    torch.isin(ids, torch.tensor(eos_ids, device=device)).all().item())

    rows, round_data, captured = [], [], []
    for round_index in range(args.rounds):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        request_started = started
        initial = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize(device)
        prefill_ms = (time.perf_counter() - started) * 1000
        cache = initial.past_key_values
        next_id = initial.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del initial
        prefill_peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        generated, round_logits = [next_id.detach()], []
        stopped_at_eos = is_eos(next_id)
        for step in range(args.warmup_steps + args.steps):
            if stopped_at_eos:
                break
            if args.feeding == "teacher":
                next_id = prompt[:, step % context_length].unsqueeze(1).contiguous()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            out = model(input_ids=next_id, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if step >= args.warmup_steps:
                rows.append({"round": round_index + 1, "step": step - args.warmup_steps + 1,
                             "input_context_tokens": context_length + step + 1,
                             "latency_ms": elapsed_ms})
                if args.save_logits:
                    round_logits.append(out.logits[:, -1, :].detach().float().cpu())
            generated.append(next_id.detach())
            stopped_at_eos = is_eos(next_id)
            del out
        request_ms = (time.perf_counter() - request_started) * 1000
        decode_peak = torch.cuda.max_memory_allocated(device)
        generated_ids = torch.cat(generated, dim=1).cpu().tolist()
        round_data.append({"round": round_index + 1, "prefill_ms": prefill_ms,
                           "request_ms_including_warmup": request_ms,
                           "generated_tokens_per_request": len(generated_ids[0]),
                           "prefill_peak_allocated_bytes": prefill_peak,
                           "decode_peak_allocated_bytes": decode_peak,
                           "predicted_token_ids": generated_ids,
                           "generated_text": tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
                           "stopped_at_eos": stopped_at_eos})
        if round_logits:
            captured.append(torch.stack(round_logits))
        del cache, generated, next_id
        print(f"completed {args.method} L={context_length} B={args.batch_size} round={round_index + 1}/{args.rounds}", flush=True)

    values = [row["latency_ms"] for row in rows]
    result = {
        "method": args.method, "model": str(Path(args.model).resolve()),
        "context_length": context_length, "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps, "steps": args.steps, "rounds": args.rounds,
        "feeding": args.feeding, "prompt_sha256": prompt_digest,
        "stop_on_eos": args.stop_on_eos, "eos_token_ids": eos_ids,
        "prompt_jsonl": str(args.prompt_jsonl.resolve()) if args.prompt_jsonl else None,
        "sample_index": args.sample_index if sample else None,
        "sample_id": sample.get("index") if sample else None,
        "reference_answers": sample.get("outputs") if sample else None,
        "dtype": args.dtype, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "config_path": str(config_path) if config_path else None, "config": config,
        "measured_steps": len(values),
        "mean_decode_step_ms": statistics.fmean(values) if values else None,
        "median_decode_step_ms": statistics.median(values) if values else None,
        "estimated_batch_tokens_per_second": args.batch_size * 1000 / statistics.fmean(values) if values else None,
        "rounds_data": round_data,
        "scope": "real sequential model forward + greedy argmax, synchronized wall time; request includes prefill and decode but excludes tokenization, detokenization and scheduling",
        "note": "One prompt is physically replicated across batch; actual checkpoint weights and full dynamic KV cache are used.",
    }
    with (output / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["round", "step", "input_context_tokens", "latency_ms"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if captured:
        torch.save(torch.stack(captured), output / "teacher_logits.pt")
    print(json.dumps({k: result[k] for k in ("method", "context_length", "batch_size", "mean_decode_step_ms", "median_decode_step_ms")}, indent=2))


if __name__ == "__main__":
    main()
