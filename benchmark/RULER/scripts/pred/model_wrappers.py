# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional

from gas_attention.pyimpl import enable_sparse_attention
from gas_attention.pyimpl.profiler import profiler


def _summarize_query_profile(
    records: List[dict], sparse_start_layer: int
) -> dict:
    metric_names = (
        "final_loaded_kv_tokens_mean",
        "final_union_amplification_mean",
        "final_per_head_selected_mean",
        "final_group_union_count_mean",
        "pre_loaded_kv_tokens_mean",
        "pre_union_amplification_mean",
        "post_loaded_kv_tokens_mean",
        "post_union_amplification_mean",
        "post_group_shared_count_mean",
    )
    sparse_records = [
        record
        for record in records
        if int(record.get("layer_id", -1)) >= sparse_start_layer
    ]
    values = {name: [] for name in metric_names}
    for record in sparse_records:
        for name in metric_names:
            value = record.get("gqa", {}).get(name)
            if isinstance(value, (int, float)):
                values[name].append(float(value))

    summary = {
        "profile_record_count_total": len(records),
        "profile_record_count": len(sparse_records),
        "metric_start_layer": sparse_start_layer,
    }
    for name, items in values.items():
        if items:
            summary[name] = sum(items) / len(items)
    summary["pg1_checked_records"] = sum(
        int(record.get("gqa", {}).get("pg1_checked_records", 0))
        for record in sparse_records
    )
    summary["pg1_violation_count"] = sum(
        int(record.get("gqa", {}).get("pg1_violation_count", 0))
        for record in sparse_records
    )
    return summary


class HuggingFaceModel:
    def __init__(
        self, name_or_path: str, algo_config_path: str, **generation_kwargs
    ) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True, use_fast=False
        )

        if "Yarn-Llama" in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        with open(algo_config_path, "r") as f:
            algo_config = json.load(f)

        self.sparse_start_layer = 2 if algo_config.get(
            "skip_first_two_layers", True
        ) else 0

        self.budget_info, self.score_info = enable_sparse_attention(
            self.pipeline.model,
            sparse_config=algo_config,
            enable_budget_info=True,
            enable_score_info=True,
        )

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        profiler.begin_query()
        try:
            if self.pipeline is None:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                    self.model.device
                )
                generated_ids = self.model.generate(**inputs, **self.generation_kwargs)
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            else:
                output = self.pipeline(
                    text_inputs=prompts,
                    **self.generation_kwargs,
                )
                assert len(output) == len(prompts)
                # output in the form of a list of list of dictionaries
                # outer list len = batch size
                # inner list len = 1
                generated_texts = [llm_result[0]["generated_text"] for llm_result in output]
        finally:
            query_profile = profiler.end_query()

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(
                    prompt, return_tensors="pt", padding=True
                )
                prompt = self.tokenizer.decode(
                    tokenized_prompt.input_ids[0], skip_special_tokens=True
                )
            if text.startswith(prompt):
                text = text[len(prompt) :]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({"text": [text]})
        
        # add budget info and score info to the results
        assert len(prompts) == 1 # only support batch size 1 for now\
        avg_budget = self.budget_info.get_total_avg_budget()
        avg_score = self.score_info.get_total_avg_score()
        self.budget_info.reset()
        self.score_info.reset()
        results[0]["budget"] = avg_budget
        results[0]["score_sum"] = avg_score
        results[0]["kv_metrics"] = _summarize_query_profile(
            query_profile, self.sparse_start_layer
        )

        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(
            name_or_path, device=self.device, dtype=torch.bfloat16
        )
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")
        self.max_genlen = self.generation_kwargs.pop("max_new_tokens")
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {"text": [self.tokenizer.decode(out.sequences[0][input_ids.shape[1] :])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
