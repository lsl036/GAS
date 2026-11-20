# This is a Python implementation of Several Approximate Attention Algorithms, hack
# the Llama

# Note that this Python version is just for testing accuracy, not for efficiency.
# Hence we use a "naive" implementation, which will compute full attention weights.

# Reference:
#   - https://github.com/mit-han-lab/Quest/blob/main/evaluation/quest_attention.py


import json
import math
import sys
from typing import Optional, Tuple, Dict
from pathlib import Path

import torch
from torch import nn

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.modeling_flash_attention_utils import _flash_attention_forward

from transformers.cache_utils import DynamicCache

from transformers.models.mistral.modeling_mistral import MistralAttention
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
except ImportError:
    Qwen3Attention = ()

# Prefill compression policies
from .snap_kv import snap_kv_compression, prepare_inputs_for_generation_llama_snapkv

# Index selector policies
from .streaming_llm import streaming_selector
from .quest import quest_selector
from .top_k import oracle_topk_selector
from .tidal_decode import tidal_decode_selector
from .sparq import sparq_selector
from .double_sparse import double_sparse_selector, init_model_channel_config

# Weight estimator policies
from .top_p import top_p_unnormalized
from .elementwise_threshold import elementwise_threshold
from .quantize import (
    min_max_per_token_quant_kv,
    max_per_token_quant_kv,
    pack_min_max_int4_k_for_gemv,
    py_int4_min_max_per_token_quant_kv,
)
from .profiler import profiler
from .gqa_refine import build_group_refined_mask, compute_gqa_union_metrics
from .gqa_fused_decode import get_gqa_fused_decode_module

# States
from .state import (
    CompressorType,
    IndexSelectorType,
    WeightEstimatorType,
    WeightPrunerType,
    LocalState,
    HistoryBudgetInfo,
    HistoryAccumulatedScoreInfo,
)

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "flash-topk-attention"))
    from ftka.cuda_ops.gemv import sparse_batched_gemv_int4_k
except ImportError:
    sparse_batched_gemv_int4_k = None


def _can_use_cuda_fused_gqa_decode(self, gqa_refinement: Dict, q_len: int) -> bool:
    return (
        q_len == 1
        and gqa_refinement.get("cuda_fused_decode", False)
        and gqa_refinement.get("enabled", False)
        and gqa_refinement.get("mode") == "group_max"
        and abs(float(gqa_refinement.get("threshold", 0.0)) - 0.80) < 1e-6
        and int(gqa_refinement.get("exception_budget", 0)) == 16
        and self.state.selector == IndexSelectorType.QUEST
        and self.state.weight_estimator == WeightEstimatorType.MIN_MAX_QUANT
        and self.state.weight_pruner == WeightPrunerType.TOP_P
        and not self.state.use_estimated_weights_in_attn
    )


# - hidden_states：输入到当前 attention 层的 hidden states，形状通常是 [batch_size, q_len, hidden_size]。prefill 时 q_len > 1，decode 时 q_len == 1。
# - attention_mask：attention mask，用来屏蔽 padding 或未来 token。我们补的 dense prefill 回退里会用它；decode 阶段代码目前要求它是 None。
# - position_ids：每个 token 的位置编号，用于 RoPE 旋转位置编码。旧版 Transformers 会传这个。
# - past_key_value：旧版 Transformers 的 KV cache 参数，保存历史 key/value，decode 时避免重复算前文。
# - output_attentions：是否返回 attention weights。默认 False，正常推理只需要输出 hidden states。
# - use_cache：是否启用 KV cache。长文本自回归 decode 基本都会是 True。
# - position_embeddings：新版 Transformers 传进来的 RoPE 结果，通常是 (cos, sin)。新版里 attention module 不再自己持有 self.rotary_emb，而是上层预先算好传进来。
# - **kwargs：兼容不同 Transformers 版本的额外参数。比如新版可能传 past_key_values，代码里会从 kwargs 里取出来兼容：

def attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    timer = profiler.new_timer()
    timer.cuda_start("total_decode_attention")
    torch.cuda.empty_cache()

    bsz, q_len, _ = hidden_states.size()

    # Shape conversion:
    #   Q: [bs, num_heads, q_len, head_dim]
    #   K: [bs, num_kv_heads, q_len, head_dim]
    #   V: [bs, num_kv_heads, q_len, head_dim]
    bs, q_len, _ = hidden_states.size()

    # print(q_len)

    timer.cuda_start("qkv_proj_rope_cache")
    query_states = self.q_proj(hidden_states).view(
        bs, q_len, self.num_heads, self.head_dim
    )
    key_states = self.k_proj(hidden_states).view(
        bs, q_len, self.num_key_value_heads, self.head_dim
    )
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = (
        self.v_proj(hidden_states)
        .view(bs, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is None and "past_key_values" in kwargs:
        past_key_value = kwargs["past_key_values"]
    elif past_key_value is None and "past_key_value" in kwargs:
        past_key_value = kwargs["past_key_value"]

    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len") and self.kv_seq_len != 0:
            kv_seq_len += self.kv_seq_len

    # Apply rotary embedding
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)
    try:
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
    except TypeError:
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

    # Concat current KV with past key value cache
    # New cache format
    # if isinstance(past_key_value, DynamicCache):
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if q_len == 1:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        else:
            # KV cache compression in prefill stage， 预填充阶段压缩KV缓存
            cache_key, cache_value = key_states, value_states
            if self.state.compressor == CompressorType.SNAP_KV:
                cache_key, cache_value = snap_kv_compression(
                    query_states,
                    key_states,
                    value_states,
                    **self.state.compressor_args,
                )
            past_key_value.update(cache_key, cache_value, self.layer_idx, cache_kwargs)
    # Legacy cache format
    # else:
    #     if past_key_value is not None:
    #         # reuse k, v, self_attention
    #         key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #         value_states = torch.cat([past_key_value[1], value_states], dim=2)
    #     past_key_value = (key_states, value_states) if use_cache else None

    # Prefill stage
    if q_len > 1:
        timer.cuda_end("qkv_proj_rope_cache")
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # Flash Attention's layout: [bs, seqlen, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=0.0,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=False,
            is_causal=self.is_causal,
            position_ids=position_ids,
            attn_implementation="flash_attention_2",
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        attn_weights = None

        return attn_output, attn_weights

    # Currently we assert Attention Mask to be None
    assert attention_mask is None

    kv_key_states = key_states
    kv_value_states = value_states

    # GQA
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    timer.cuda_end("qkv_proj_rope_cache")

    # Computate full attention weights, which is OK (will not OOM) for decode
    timer.cuda_start("full_attn_logits")
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )
    timer.cuda_end("full_attn_logits")

    # torch.set_printoptions(profile="full")
    # print(f"Layer {self.layer_id}, Real Weights: ", attn_weights)

    # 跳过前两层，从第三层开始进行稀疏化
    start_layer = 2 if self.state.skip_first_two_layers else 0

    max_selected = 0
    avg_selected = 0.0
    gqa_metrics = {}
    fused_attn_output = None
    final_metric_mask = torch.ones_like(attn_weights, dtype=torch.bool)
    if self.layer_id >= start_layer:
        # Index Selector
        timer.cuda_start("selector")
        if self.state.selector == IndexSelectorType.NONE:
            mask = torch.ones_like(attn_weights, dtype=torch.bool)
        elif self.state.selector == IndexSelectorType.STREAMING:
            mask = streaming_selector(
                attn_weights,
                self.state.selector_args["token_budget"],
                self.state.selector_args["num_sinks"],
            )
        elif self.state.selector == IndexSelectorType.QUEST:
            mask = quest_selector(
                self,
                query_states,
                key_states,
                **self.state.selector_args,
            )
        elif self.state.selector == IndexSelectorType.ORACLE_TOPK:
            mask = oracle_topk_selector(
                attn_weights,
                self.state.selector_args["token_budget"],
            )
        elif self.state.selector == IndexSelectorType.TIDAL_DECODE:
            mask = tidal_decode_selector(
                self.state,
                attn_weights,
                self.state.selector_args["token_budget"],
                self.layer_id,
                self.state.selector_args["reselection_layers"],
            )
        elif self.state.selector == IndexSelectorType.SPARQ:
            mask = sparq_selector(
                query_states,
                key_states,
                **self.state.selector_args,
            )
        elif self.state.selector == IndexSelectorType.DS:
            selector_args = self.state.selector_args.copy()
            selector_args.pop("config_path")
            selector_args.pop("selected_channel")
            mask = double_sparse_selector(
                query_states,
                key_states,
                self.sorted_channel,
                **selector_args,
            )
        timer.cuda_end("selector")

        # Weight Estimator
        if self.state.weight_estimator == WeightEstimatorType.NONE:
            estimated_weights = attn_weights
        elif self.state.weight_estimator == WeightEstimatorType.MIN_MAX_QUANT:
            timer.cuda_start("fake_quant_estimator")
            quantized_key = min_max_per_token_quant_kv(
                key_states,
                self.state.weight_estimator_args["quant_bit"],
            )
            estimated_weights = torch.matmul(
                query_states, quantized_key.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            # 将未选中的token的权重设置为极小值
            estimated_weights[~mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
            timer.cuda_end("fake_quant_estimator")
        elif self.state.weight_estimator == WeightEstimatorType.PY_INT4_MIN_MAX_QUANT:
            timer.cuda_start("py_int4_quant")
            quantized_key, quant_scales, quant_zeros = py_int4_min_max_per_token_quant_kv(
                key_states,
                self.state.weight_estimator_args["quant_bit"],
            )
            timer.cuda_end("py_int4_quant")
            timer.cuda_start("py_int4_dequant_matmul")
            dequant_key = quantized_key.to(query_states.dtype) * quant_scales + quant_zeros
            estimated_weights = torch.matmul(
                query_states, dequant_key.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            estimated_weights[~mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
            timer.cuda_end("py_int4_dequant_matmul")
        elif self.state.weight_estimator == WeightEstimatorType.REAL_INT4_MIN_MAX_QUANT:
            if sparse_batched_gemv_int4_k is None:
                raise ImportError("ftka.cuda_ops.gemv.sparse_batched_gemv_int4_k is required")
            quant_bit = self.state.weight_estimator_args["quant_bit"]
            if quant_bit != 4:
                raise ValueError("REAL_INT4_MIN_MAX_QUANT only supports quant_bit=4")

            timer.cuda_start("int4_pack")
            k_cache_int4, quant_scales, quant_zeros = pack_min_max_int4_k_for_gemv(
                key_states
            )
            timer.cuda_end("int4_pack")
            seq_len = key_states.shape[-2]
            timer.wall_start("mask_to_selected_indices")
            candidate_counts = mask.squeeze(2).sum(dim=-1)
            max_selected = int(candidate_counts.max().item())
            avg_selected = float(candidate_counts.float().mean().item())
            selected_indices = torch.empty(
                (bsz, self.num_heads, max_selected),
                dtype=torch.int32,
                device=query_states.device,
            )
            selected_valid = torch.zeros(
                (bsz, self.num_heads, max_selected),
                dtype=torch.uint8,
                device=query_states.device,
            )
            mask_2d = mask.squeeze(2)
            for batch_idx in range(bsz):
                for head_idx in range(self.num_heads):
                    indices = torch.nonzero(
                        mask_2d[batch_idx, head_idx], as_tuple=False
                    ).flatten()
                    count = indices.numel()
                    if count > 0:
                        selected_indices[batch_idx, head_idx, :count] = indices.to(
                            torch.int32
                        )
                        selected_valid[batch_idx, head_idx, :count] = 1
                    if count < max_selected:
                        selected_indices[batch_idx, head_idx, count:] = 0
            timer.wall_end("mask_to_selected_indices")

            sparse_gemv_out = torch.empty(
                (bsz, self.num_heads, max_selected),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            timer.cuda_start("sparse_int4_gemv")
            sparse_batched_gemv_int4_k(
                query_states.squeeze(2).contiguous(),
                sparse_gemv_out,
                k_cache_int4,
                quant_scales,
                quant_zeros,
                selected_indices,
                selected_valid,
            )
            timer.cuda_end("sparse_int4_gemv")
            timer.wall_start("scatter_sparse_logits")
            gemv_out = torch.zeros(
                (bsz, self.num_heads, seq_len),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            for batch_idx in range(bsz):
                for head_idx in range(self.num_heads):
                    count = int(candidate_counts[batch_idx, head_idx].item())
                    if count > 0:
                        gemv_out[
                            batch_idx,
                            head_idx,
                            selected_indices[batch_idx, head_idx, :count].long(),
                        ] = sparse_gemv_out[batch_idx, head_idx, :count]
            timer.wall_end("scatter_sparse_logits")
            timer.cuda_start("estimated_logits_finalize")
            estimated_weights = gemv_out.unsqueeze(2) / math.sqrt(self.head_dim)
            estimated_weights[~mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
            timer.cuda_end("estimated_logits_finalize")
        elif self.state.weight_estimator == WeightEstimatorType.MAX_QUANT:
            timer.cuda_start("fake_quant_estimator")
            quantized_key = max_per_token_quant_kv(
                key_states,
                self.state.weight_estimator_args["quant_bit"],
                self.state.weight_estimator_args["smooth_k"],
            )
            estimated_weights = torch.matmul(
                query_states, quantized_key.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            # 将未选中的token的权重设置为极小值
            estimated_weights[~mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
            timer.cuda_end("fake_quant_estimator")

        # Prune the weights
        # The input is un-normalized
        timer.cuda_start("top_p_prune")
        if self.state.weight_pruner == WeightPrunerType.THRESHOLD:
            pruned_mask = elementwise_threshold(
                estimated_weights,
                self.state.weight_pruner_args["threshold"],
            )
        elif self.state.weight_pruner == WeightPrunerType.TOP_P:
            pruned_mask = top_p_unnormalized(
                estimated_weights,
                self.state.weight_pruner_args["threshold"],
            )

        if self.state.weight_pruner != WeightPrunerType.NONE:
            mask = mask & pruned_mask
        timer.cuda_end("top_p_prune")

        gqa_refinement = getattr(self.state, "gqa_refinement", {}) or {}
        if gqa_refinement.get("metrics_only", False) or gqa_refinement.get(
            "enabled", False
        ):
            timer.wall_start("gqa_union_metrics_pre")
            pre_metrics = compute_gqa_union_metrics(mask, self.num_key_value_heads)
            timer.wall_end("gqa_union_metrics_pre")
            gqa_metrics.update(pre_metrics)
            gqa_metrics.update(
                {f"pre_{key}": value for key, value in pre_metrics.items()}
            )

        if gqa_refinement.get("enabled", False):
            mode = gqa_refinement["mode"]
            threshold = gqa_refinement.get(
                "threshold",
                self.state.weight_pruner_args.get("threshold", 0.85),
            )
            pre_gqa_mask = mask
            use_cuda_fused_decode = _can_use_cuda_fused_gqa_decode(
                self, gqa_refinement, q_len
            )
            if use_cuda_fused_decode:
                timer.cuda_start("gqa_refine")
                shared_mask = build_group_refined_mask(
                    estimated_weights,
                    mask,
                    self.num_key_value_heads,
                    threshold,
                    mode,
                    0,
                )
                mask = build_group_refined_mask(
                    estimated_weights,
                    mask,
                    self.num_key_value_heads,
                    threshold,
                    mode,
                    gqa_refinement.get("exception_budget", 0),
                )
                timer.cuda_end("gqa_refine")

                timer.cuda_start("cuda_compact_indices")
                fused_module = get_gqa_fused_decode_module()
                max_group_selected = max(
                    1, int(shared_mask.squeeze(2).sum(dim=-1).max().item())
                )
                compact_outputs = fused_module.compact_group_exception_masks(
                    shared_mask.contiguous(),
                    mask.contiguous(),
                    int(self.num_key_value_heads),
                    int(max_group_selected),
                    int(gqa_refinement.get("exception_budget", 0)),
                )
                group_indices, exception_indices, group_counts, exception_counts = (
                    compact_outputs
                )
                timer.cuda_end("cuda_compact_indices")

                timer.cuda_start("fused_group_exception_decode")
                fused_attn_output = fused_module.batched_gqa_group_exception_decode(
                    query_states.squeeze(2).contiguous(),
                    kv_key_states.contiguous(),
                    kv_value_states.contiguous(),
                    group_indices.contiguous(),
                    exception_indices.contiguous(),
                )
                timer.cuda_end("fused_group_exception_decode")

                gqa_metrics["fused_group_selected_mean"] = float(
                    group_counts.float().mean().item()
                )
                gqa_metrics["fused_exception_selected_mean"] = float(
                    exception_counts.float().mean().item()
                )
            else:
                timer.cuda_start("gqa_refine")
                mask = build_group_refined_mask(
                    estimated_weights,
                    mask,
                    self.num_key_value_heads,
                    threshold,
                    mode,
                    gqa_refinement.get("exception_budget", 0),
                )
                timer.cuda_end("gqa_refine")

            if gqa_refinement.get("pg1_audit", False):
                if abs(float(threshold) - 1.0) >= 1e-6:
                    raise ValueError("pg1_audit requires gqa_refinement.threshold=1.0")
                grouped_pre_mask = pre_gqa_mask.reshape(
                    bsz,
                    self.num_key_value_heads,
                    self.num_heads // self.num_key_value_heads,
                    1,
                    pre_gqa_mask.shape[-1],
                )
                expected_pg1_mask = (
                    grouped_pre_mask.any(dim=2)
                    .unsqueeze(2)
                    .expand_as(
                        grouped_pre_mask
                    )
                    .reshape_as(pre_gqa_mask)
                )
                violation = int(not torch.equal(mask, expected_pg1_mask))
                gqa_metrics["pg1_checked_records"] = 1
                gqa_metrics["pg1_violation_count"] = violation
                if violation:
                    raise RuntimeError(
                        "PG1 audit failed: refined mask is not the exact GQA union"
                    )

            timer.wall_start("gqa_union_metrics_post")
            post_metrics = compute_gqa_union_metrics(mask, self.num_key_value_heads)
            timer.wall_end("gqa_union_metrics_post")
            gqa_metrics.update(
                {f"post_{key}": value for key, value in post_metrics.items()}
            )

        # Record budget
        timer.cuda_start("budget_score_update")
        if self.state.budget_info is not None:
            self.state.budget_info.update(self.layer_id, mask)

            # if self.layer_id == 31:
            #     # print(self.state.budget_info.get_avg_budget_cur_query())
            #     self.state.budget_info.print_budget_info_single_query_with_B0() / 0

        # Record the score
        if self.state.score_info is not None:
            self.state.score_info.update(self.layer_id, attn_weights, mask)
        timer.cuda_end("budget_score_update")

        if self.state.use_estimated_weights_in_attn:
            attn_weights = estimated_weights

        # Mask unselected tokens
        timer.cuda_start("final_mask")
        attn_weights[~mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
        timer.cuda_end("final_mask")
        final_metric_mask = mask

    timer.wall_start("gqa_union_metrics_final")
    final_metrics = compute_gqa_union_metrics(
        final_metric_mask, self.num_key_value_heads
    )
    timer.wall_end("gqa_union_metrics_final")
    gqa_metrics.update({f"final_{key}": value for key, value in final_metrics.items()})

    if fused_attn_output is None:
        # upcast attention to fp32
        timer.cuda_start("softmax_value")
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        attn_output = torch.matmul(attn_weights, value_states)
        timer.cuda_end("softmax_value")
    else:
        attn_output = fused_attn_output.unsqueeze(2)

    # Evaluate cos similarity

    # attn_weights_real = torch.matmul(
    #     query_states, key_states.transpose(2, 3)
    # ) / math.sqrt(self.head_dim)
    # attn_weights_real = nn.functional.softmax(
    #     attn_weights_real, dim=-1, dtype=torch.float32
    # ).to(query_states.dtype)
    # attn_output_real = torch.matmul(attn_weights_real, value_states)

    # print(
    #     "avg cos sim: ",
    #     nn.CosineSimilarity(dim=-1, eps=1e-6)(attn_output, attn_output_real)
    #     .flatten()
    #     .mean(),
    # )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    timer.cuda_start("o_proj")
    attn_output = self.o_proj(attn_output)
    timer.cuda_end("o_proj")

    if not output_attentions:
        attn_weights = None

    timer.cuda_end("total_decode_attention")
    timer.finalize()
    profiler.record(
        {
            "layer_id": self.layer_id,
            "seq_len": int(key_states.shape[-2]),
            "num_heads": int(self.num_heads),
            "max_selected": int(max_selected),
            "avg_selected": float(avg_selected),
            "gqa": gqa_metrics,
            "stage_times_us": timer.stage_times_us,
            "total_us": timer.stage_times_us.get("total_decode_attention", 0.0),
        }
    )

    return attn_output, attn_weights


global local_state
local_state = LocalState()


"""
Enable Sparse Attention through a monkey patching way.
"""


def enable_sparse_attention(
    model,
    sparse_config: Dict,
    enable_budget_info: bool = False,
    enable_score_info: bool = False,
) -> (Optional[HistoryBudgetInfo], Optional[HistoryAccumulatedScoreInfo]):
    """
    Enable Sparse Attention via config dict.

    Config template:
    {
        "selector": {
            "type": "type_name",
            ... // Other args
        }
        "weight_estimator": {
            "type": "type_name",
            ... // Other args
        }
        "weight_pruner": {
            "type": "type_name",
            ... // Other args
        }
        "use_estimated_weights_in_attn": False,
        "skip_first_two_layers": True,
    }
    """

    def _split_type_args(config: Dict) -> Tuple:
        config1 = config.copy()
        type_name = config1["type"]
        del config1["type"]
        return type_name, config1

    compressor_type, compressor_args = _split_type_args(sparse_config["compressor"])
    selector_type, selector_args = _split_type_args(sparse_config["selector"])
    weight_estimator_type, weight_estimator_args = _split_type_args(
        sparse_config["weight_estimator"]
    )
    weight_pruner_type, weight_pruner_args = _split_type_args(
        sparse_config["weight_pruner"]
    )

    budget_info = HistoryBudgetInfo() if enable_budget_info else None
    score_info = HistoryAccumulatedScoreInfo() if enable_score_info else None

    use_estimated_weights_in_attn = sparse_config.get(
        "use_estimated_weights_in_attn", False
    )
    skip_first_two_layers = sparse_config.get("skip_first_two_layers", True)
    gqa_refinement = sparse_config.get("gqa_refinement", {})

    _enable_sparse_attention(
        model,
        compressor=CompressorType.from_str(compressor_type),
        compressor_args=compressor_args,
        selector=IndexSelectorType.from_str(selector_type),
        selector_args=selector_args,
        weight_estimator=WeightEstimatorType.from_str(weight_estimator_type),
        weight_estimator_args=weight_estimator_args,
        weight_pruner=WeightPrunerType.from_str(weight_pruner_type),
        weight_pruner_args=weight_pruner_args,
        budget_info=budget_info,
        score_info=score_info,
        use_estimated_weights_in_attn=use_estimated_weights_in_attn,
        skip_first_two_layers=skip_first_two_layers,
        gqa_refinement=gqa_refinement,
    )

    return budget_info, score_info


def _enable_sparse_attention(
    model,
    # KV cache compression in prefill stage
    compressor: CompressorType = CompressorType.NONE,
    compressor_args: Dict = {},
    # Sparse attention index selector
    selector: IndexSelectorType = IndexSelectorType.NONE,
    selector_args: Dict = {},
    # Weight estimator and pruner
    weight_estimator: WeightEstimatorType = WeightEstimatorType.NONE,
    weight_estimator_args: Dict = {},
    weight_pruner: WeightPrunerType = WeightPrunerType.NONE,
    weight_pruner_args: Dict = {},
    # Global Args
    budget_info: Optional[HistoryBudgetInfo] = None,
    score_info: Optional[HistoryAccumulatedScoreInfo] = None,
    # Whether to use estimated weights as real weights
    use_estimated_weights_in_attn: bool = False,
    # To ensure a fair comparsion, we skip the first two layers
    skip_first_two_layers: bool = True,
    # Optional Method B GQA metrics/refinement config.
    gqa_refinement: Dict = {},
) -> None:
    """Enable Sparse Attention.

    Can't be called twice since the local_state isn't reset.
    """

    if not local_state.arg_set:
        local_state.set_args(
            compressor=compressor,
            compressor_args=compressor_args,
            selector=selector,
            selector_args=selector_args,
            weight_estimator=weight_estimator,
            weight_estimator_args=weight_estimator_args,
            weight_pruner=weight_pruner,
            weight_pruner_args=weight_pruner_args,
            budget_info=budget_info,
            score_info=score_info,
            use_estimated_weights_in_attn=use_estimated_weights_in_attn,
            skip_first_two_layers=skip_first_two_layers,
            gqa_refinement=gqa_refinement,
        )

        if selector == IndexSelectorType.DS:
            config_path = selector_args["config_path"]

            with open(config_path, "r") as f:
                channel_config = json.load(f)

            model = init_model_channel_config(
                model,
                channel_config,
                heavy_channel_num=selector_args["r"],
                selected_channel=selector_args["selected_channel"],
            )

        print(f"Local state args: {local_state.__dict__}")

    # Some monkey patching
    # if compressor == CompressorType.SNAP_KV:
    #     model.prepare_inputs_for_generation = types.MethodType(
    #         prepare_inputs_for_generation_llama_snapkv, model
    #     )

    for name, module in model.named_modules():
        if isinstance(module, (LlamaAttention, MistralAttention, Qwen3Attention)):
            if hasattr(module, "config"):
                module.num_heads = getattr(
                    module, "num_heads", module.config.num_attention_heads
                )
                module.num_key_value_heads = getattr(
                    module, "num_key_value_heads", module.config.num_key_value_heads
                )
                module.hidden_size = getattr(
                    module, "hidden_size", module.config.hidden_size
                )
                module.num_key_value_groups = getattr(
                    module,
                    "num_key_value_groups",
                    module.num_heads // module.num_key_value_heads,
                )
            # For longchat model
            # print(name)
            layer_id = int(name.split(".")[2])
            module.layer_id = layer_id

            # Hack the forward function  补丁替换 forward 函数
            module.forward = types.MethodType(attention_forward, module)

            module.state = local_state


def reset_sparse_config() -> None:
    global local_state
    local_state.arg_set = False
