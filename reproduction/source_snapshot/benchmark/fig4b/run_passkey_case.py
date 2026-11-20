#!/usr/bin/env python3
"""Run one Fig. 4(b) model/length/strategy cell on the shared cases."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from twilight.pyimpl import enable_sparse_attention  # noqa: E402
from twilight.pyimpl.profiler import profiler  # noqa: E402


GARBAGE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
TASK_DESCRIPTION = (
    "There is an important info hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about the important information there."
)
QUESTION = "What is the pass key? The pass key is"


def render_prompt(tokenizer, user_text: str, model_path: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "Qwen3" in model_path:
            kwargs["enable_thinking"] = False
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    return user_text


def user_text(pass_key: int, depth_percent: int, garbage_chars: int) -> str:
    garbage = (GARBAGE * ((garbage_chars // len(GARBAGE)) + 2))[:garbage_chars]
    split = int(garbage_chars * depth_percent / 100)
    return "\n".join(
        [
            TASK_DESCRIPTION,
            garbage[:split],
            f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key.",
            garbage[split:],
            QUESTION,
        ]
    )


def make_prompt(tokenizer, model_path: str, case: dict, target_tokens: int) -> tuple[str, int]:
    low, high = 0, max(1024, target_tokens * 8)
    for _ in range(24):
        probe = render_prompt(
            tokenizer,
            user_text(case["pass_key"], case["depth_percent"], high),
            model_path,
        )
        if len(tokenizer(probe, add_special_tokens=False).input_ids) >= target_tokens:
            break
        high *= 2
    for _ in range(24):
        middle = (low + high) // 2
        prompt = render_prompt(
            tokenizer,
            user_text(case["pass_key"], case["depth_percent"], middle),
            model_path,
        )
        length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if length < target_tokens:
            low = middle + 1
        else:
            high = middle
    prompt = render_prompt(
        tokenizer,
        user_text(case["pass_key"], case["depth_percent"], high),
        model_path,
    )
    token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return prompt, token_count


def load_prompt_cache(tokenizer, model_path: str, manifest: dict, target_tokens: int, path: Path) -> list[dict]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["cases"]
    cases = []
    for case in manifest["cases"]:
        prompt, token_count = make_prompt(tokenizer, model_path, case, target_tokens)
        cases.append({**case, "prompt": prompt, "actual_input_tokens": token_count})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"target_tokens": target_tokens, "cases": cases}, indent=2) + "\n", encoding="utf-8")
    return cases


def predict(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    question_tokens = min(128, max(1, ids.shape[-1] // 4))
    context_ids = ids[:, :-question_tokens]
    question_ids = ids[:, -question_tokens:]
    with torch.no_grad():
        output = model(input_ids=context_ids, past_key_values=None, use_cache=True)
        past = output.past_key_values
        for token in question_ids[0]:
            output = model(
                input_ids=token.view(1, 1), past_key_values=past, use_cache=True
            )
            past = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1).view(1, 1)
        generated = [int(next_token.item())]
        for _ in range(max_new_tokens - 1):
            output = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = output.past_key_values
            next_token = output.logits[:, -1, :].argmax(dim=-1).view(1, 1)
            generated.append(int(next_token.item()))
            if generated[-1] == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    profile_path = args.output_dir / "profile.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    if predictions_path.is_file() and profile_path.is_file():
        print(f"skip complete cell: {args.output_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = load_prompt_cache(
        tokenizer, args.model, manifest, args.target_tokens, args.prompt_cache
    )
    budget_info, score_info = enable_sparse_attention(
        model, sparse_config=config, enable_budget_info=True, enable_score_info=True
    )

    predictions_path.write_text("", encoding="utf-8")
    profile_path.write_text("", encoding="utf-8")
    correct = 0
    with predictions_path.open("a", encoding="utf-8") as predictions, profile_path.open(
        "a", encoding="utf-8"
    ) as profiles:
        for case in cases:
            profiler.begin_query()
            response = predict(model, tokenizer, case["prompt"], args.max_new_tokens)
            query_records = profiler.end_query()
            match = re.search(r"\d+", response)
            answer = int(match.group()) if match else response[:40]
            is_correct = answer == case["pass_key"]
            correct += int(is_correct)
            predictions.write(
                json.dumps(
                    {
                        "case_id": case["case_id"],
                        "depth_percent": case["depth_percent"],
                        "pass_key": case["pass_key"],
                        "answer": answer,
                        "response": response,
                        "correct": is_correct,
                        "actual_input_tokens": case["actual_input_tokens"],
                        "budget": budget_info.get_total_avg_budget(),
                        "score_sum": score_info.get_total_avg_score(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            budget_info.reset()
            score_info.reset()
            for record in query_records:
                profiles.write(
                    json.dumps(
                        {
                            **record,
                            "case_id": case["case_id"],
                            "depth_percent": case["depth_percent"],
                            "actual_input_tokens": case["actual_input_tokens"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    metadata_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "model_label": args.model_label,
                "config": str(args.config.resolve()),
                "config_json": config,
                "manifest": str(args.manifest.resolve()),
                "target_tokens": args.target_tokens,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "n_expected": len(cases),
                "n_scored": len(cases),
                "correct": correct,
                "accuracy": correct / len(cases) if cases else 0.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"completed {args.model_label} {args.target_tokens}: {correct}/{len(cases)}")


if __name__ == "__main__":
    main()
