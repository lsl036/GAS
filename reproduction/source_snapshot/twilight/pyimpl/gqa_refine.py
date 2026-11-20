"""GQA group-level metrics for Method B experiments."""

from typing import Dict

import torch

from .top_p import top_p_unnormalized


def _top_p_probability_mask(probabilities: torch.Tensor, threshold: float) -> torch.Tensor:
    """Select the smallest descending prefix whose mass reaches ``threshold``."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"top-p threshold must be in (0, 1], got {threshold}")

    sorted_probs, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    keep_sorted = (cumulative - sorted_probs) < threshold
    keep_sorted[..., 0] = True
    keep = torch.zeros_like(keep_sorted, dtype=torch.bool)
    keep.scatter_(-1, sorted_indices, keep_sorted)
    return keep


@torch.no_grad()
def compute_gqa_union_metrics(
    mask: torch.Tensor,
    num_key_value_heads: int,
) -> Dict[str, float]:
    """Measure how much per-query-head masks amplify after GQA union.

    Args:
        mask: Boolean attention mask with shape [bs, num_heads, 1, seq_len].
        num_key_value_heads: Number of KV heads before repeat_kv.
    """

    if mask.dim() != 4 or mask.shape[2] != 1:
        raise ValueError(f"Expected mask shape [bs, heads, 1, seq], got {mask.shape}")

    batch_size, num_heads, _, seq_len = mask.shape
    if num_key_value_heads <= 0 or num_heads % num_key_value_heads != 0:
        raise ValueError(
            f"Invalid GQA layout: num_heads={num_heads}, "
            f"num_key_value_heads={num_key_value_heads}"
        )

    group_size = num_heads // num_key_value_heads
    grouped_mask = mask.squeeze(2).reshape(
        batch_size, num_key_value_heads, group_size, seq_len
    )

    per_head_counts = grouped_mask.sum(dim=-1).float()
    union_counts = grouped_mask.any(dim=2).sum(dim=-1).float()
    shared_counts = grouped_mask.all(dim=2).sum(dim=-1).float()
    per_group_head_mean = per_head_counts.mean(dim=-1)
    amplification = union_counts / per_group_head_mean.clamp(min=1.0)
    loaded_kv_tokens = union_counts.sum(dim=-1)

    return {
        "num_kv_heads": float(num_key_value_heads),
        "group_size": float(group_size),
        "per_head_selected_mean": float(per_head_counts.mean().item()),
        "per_head_selected_max": float(per_head_counts.max().item()),
        "group_union_count_mean": float(union_counts.mean().item()),
        "group_union_count_max": float(union_counts.max().item()),
        "group_shared_count_mean": float(shared_counts.mean().item()),
        "union_amplification_mean": float(amplification.mean().item()),
        "union_amplification_max": float(amplification.max().item()),
        "loaded_kv_tokens_mean": float(loaded_kv_tokens.mean().item()),
        "loaded_kv_tokens_max": float(loaded_kv_tokens.max().item()),
    }


@torch.no_grad()
def build_group_shared_mask(
    estimated_weights: torch.Tensor,
    candidate_mask: torch.Tensor,
    num_key_value_heads: int,
    threshold: float,
    mode: str,
) -> torch.Tensor:
    """Build one shared token mask for each GQA group.

    The returned mask is broadcast back to query heads, so all query heads mapped to
    the same KV head use the same selected tokens.
    """

    if estimated_weights.shape != candidate_mask.shape:
        raise ValueError(
            "estimated_weights and candidate_mask must have the same shape, got "
            f"{estimated_weights.shape} and {candidate_mask.shape}"
        )

    batch_size, num_heads, _, seq_len = candidate_mask.shape
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            f"Invalid GQA layout: num_heads={num_heads}, "
            f"num_key_value_heads={num_key_value_heads}"
        )

    group_size = num_heads // num_key_value_heads
    grouped_scores = estimated_weights.reshape(
        batch_size, num_key_value_heads, group_size, 1, seq_len
    )
    grouped_mask = candidate_mask.reshape(
        batch_size, num_key_value_heads, group_size, 1, seq_len
    )
    union_mask = grouped_mask.any(dim=2)

    # p_g=1 is the audit boundary for GAS: retain the complete union exactly
    # instead of relying on floating-point top-p accumulation to keep it.
    if threshold == 1.0:
        group_mask = union_mask
        return (
            group_mask.unsqueeze(2)
            .expand(batch_size, num_key_value_heads, group_size, 1, seq_len)
            .reshape(batch_size, num_heads, 1, seq_len)
        )

    if mode == "group_max":
        min_value = torch.finfo(estimated_weights.dtype).min
        group_scores = grouped_scores.masked_fill(~grouped_mask, min_value).max(
            dim=2
        ).values
    elif mode == "group_mean":
        valid_count = grouped_mask.sum(dim=2).clamp(min=1)
        group_scores = (grouped_scores * grouped_mask).sum(dim=2) / valid_count
    elif mode == "prob_sum":
        # Normalize each query head over its own candidate set first, then
        # average those distributions inside the GQA group. This is a
        # probability-space ablation and must not call top_p_unnormalized,
        # which applies another softmax to logits.
        score_dtype = torch.float32
        masked_scores = grouped_scores.to(score_dtype).masked_fill(
            ~grouped_mask, torch.finfo(score_dtype).min
        )
        head_probs = torch.softmax(masked_scores, dim=-1) * grouped_mask
        active_heads = grouped_mask.any(dim=(-2, -1)).sum(dim=2).clamp(min=1)
        group_probs = head_probs.sum(dim=2) / active_heads.unsqueeze(-1).unsqueeze(-1)
        group_probs = group_probs.masked_fill(~union_mask, 0.0)
        group_mask = _top_p_probability_mask(group_probs, threshold) & union_mask
        return (
            group_mask.unsqueeze(2)
            .expand(batch_size, num_key_value_heads, group_size, 1, seq_len)
            .reshape(batch_size, num_heads, 1, seq_len)
        )
    else:
        raise ValueError(f"Unsupported GQA refinement mode: {mode}")

    group_scores = group_scores.masked_fill(
        ~union_mask, torch.finfo(group_scores.dtype).min
    )
    group_mask = top_p_unnormalized(group_scores, threshold) & union_mask
    return (
        group_mask.unsqueeze(2)
        .expand(batch_size, num_key_value_heads, group_size, 1, seq_len)
        .reshape(batch_size, num_heads, 1, seq_len)
    )


@torch.no_grad()
def add_per_head_exception_mask(
    estimated_weights: torch.Tensor,
    candidate_mask: torch.Tensor,
    shared_mask: torch.Tensor,
    exception_budget: int,
) -> torch.Tensor:
    """Add per-head exception tokens outside the group-shared mask."""

    if exception_budget <= 0:
        return shared_mask
    if estimated_weights.shape != candidate_mask.shape:
        raise ValueError(
            "estimated_weights and candidate_mask must have the same shape, got "
            f"{estimated_weights.shape} and {candidate_mask.shape}"
        )
    if shared_mask.shape != candidate_mask.shape:
        raise ValueError(
            f"shared_mask shape {shared_mask.shape} does not match "
            f"candidate_mask shape {candidate_mask.shape}"
        )

    residual_mask = candidate_mask & ~shared_mask
    if not residual_mask.any():
        return shared_mask

    k = min(exception_budget, estimated_weights.shape[-1])
    min_value = torch.finfo(estimated_weights.dtype).min
    residual_scores = estimated_weights.masked_fill(~residual_mask, min_value)
    topk_indices = torch.topk(residual_scores, k=k, dim=-1).indices
    topk_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    topk_mask.scatter_(-1, topk_indices, True)
    return shared_mask | (topk_mask & residual_mask)


@torch.no_grad()
def build_group_refined_mask(
    estimated_weights: torch.Tensor,
    candidate_mask: torch.Tensor,
    num_key_value_heads: int,
    threshold: float,
    mode: str,
    exception_budget: int = 0,
) -> torch.Tensor:
    """Build a group-shared mask with optional per-head exception tokens."""

    shared_mask = build_group_shared_mask(
        estimated_weights,
        candidate_mask,
        num_key_value_heads,
        threshold,
        mode,
    )
    return add_per_head_exception_mask(
        estimated_weights,
        candidate_mask,
        shared_mask,
        exception_budget,
    )
