"""Connected, fixed-trace attention paths for the repaired E7 experiment.

This is a component replay, not a serving integration. The quality-evaluation
modules are called without patching the model or modifying their source.
"""

from contextlib import contextmanager
import math
from types import SimpleNamespace

import torch


@contextmanager
def stage(records, name):
    if records is None:
        yield
        return
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    yield
    end.record()
    records.append((name, begin, end))


def rms_norm(x, weight, eps):
    value = x.float()
    return weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def quest_candidates(q, minimum, maximum, length, budget=8192, chunk=16, pages_only=False):
    batch, heads, dim = q.shape
    kv_heads = minimum.shape[1]
    group = heads // kv_heads
    query = q.view(batch, kv_heads, group, dim)
    # One GEMM avoids rounding two separately accumulated partial scores.
    query = torch.cat((query.clamp(min=0), query.clamp(max=0)), dim=-1)
    bounds = torch.cat((maximum, minimum), dim=-1)
    scores = torch.matmul(query, bounds.transpose(-1, -2))
    count = min(math.ceil(budget / chunk), scores.shape[-1])
    pages = scores.topk(count, dim=-1).indices
    if pages_only:
        return pages.reshape(batch, heads, -1).int()
    tokens = (pages[..., None] * chunk + torch.arange(chunk, device=q.device)).reshape(batch, heads, -1)
    mask = torch.zeros(batch, heads, length, dtype=torch.bool, device=q.device)
    mask.scatter_(-1, tokens, True)
    return mask


def mask_to_pages(mask, length, group_factor):
    """Compact boolean supports to FlashInfer metadata without copying KV."""
    flat = mask.reshape(-1, length)
    rows, tokens = flat.nonzero(as_tuple=True)
    counts = flat.sum(-1, dtype=torch.int32)
    indptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    ids = ((rows // group_factor) * length + tokens).int()
    last = torch.ones(flat.shape[0], dtype=torch.int32, device=mask.device)
    return indptr, ids, last


class AttentionPath:
    def __init__(self, method, q, k, v, module, budget=8192):
        from twilight.pyimpl.quantize import min_max_per_token_quant_kv
        from twilight.pyimpl.top_p import top_p_unnormalized
        from twilight.pyimpl.gqa_refine import build_group_shared_mask, add_per_head_exception_mask

        self.method, self.q, self.k, self.v = method, q, k, v
        self.module, self.budget = module, min(budget, k.shape[-2])
        self.batch, self.heads, self.dim = q.shape
        self.kv_heads, self.length = k.shape[1:3]
        self.group = self.heads // self.kv_heads
        if self.length % 16:
            raise ValueError("E7 v2 requires lengths divisible by the frozen chunk size 16")
        self.quant, self.top_p = min_max_per_token_quant_kv, top_p_unnormalized
        self.shared_fn, self.exception_fn = build_group_shared_mask, add_per_head_exception_mask
        self.new_k, self.new_v = k[:, :, -1].clone(), v[:, :, -1].clone()
        self.output = torch.empty_like(q)
        self.last = {}
        self.dispatch = method
        if method == "fa2":
            from flash_attn import flash_attn_with_kvcache
            self.fa2 = flash_attn_with_kvcache
            # This is the backend's persistent cache layout, not a per-step copy.
            self.fa_k = k.transpose(1, 2).contiguous()
            self.fa_v = v.transpose(1, 2).contiguous()
        else:
            import flashinfer
            workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
            self.pages = (k.view(-1, 1, 1, self.dim), v.view(-1, 1, 1, self.dim))
            if method == "flashinfer":
                # Dense prefix page addresses are query-independent. Only
                # sequence-length metadata/plan needs per-step updating.
                self.fi_k = k.transpose(1, 2).contiguous()
                self.fi_v = v.transpose(1, 2).contiguous()
                self.pages = (self.fi_k.view(-1, 16, self.kv_heads, self.dim), self.fi_v.view(-1, 16, self.kv_heads, self.dim))
                requests, pages_per_request = self.batch, self.length // 16
                self.dense_metadata = (
                    torch.arange(requests + 1, device=q.device, dtype=torch.int32) * pages_per_request,
                    torch.arange(requests * pages_per_request, device=q.device, dtype=torch.int32),
                    torch.full((requests,), 16, device=q.device, dtype=torch.int32),
                )
            elif method == "quest":
                self.pages = (k.view(-1, 16, 1, self.dim), v.view(-1, 16, 1, self.dim))
        if method in ("quest", "quest_twi", "gas"):
            blocks = k.view(self.batch, self.kv_heads, -1, 16, self.dim)
            self.minimum, self.maximum = blocks.amin(3), blocks.amax(3)
        if method in ("quest_twi", "gas"):
            self.khat = torch.empty_like(k)
            # Key-only quantization is query-independent and persistent. Only
            # the appended token is requantized in each timed replay.
            for start in range(0, self.length, 256):
                self.khat[:, :, start:start + 256] = self.quant(k[:, :, start:start + 256], 4)

    def plan_and_run(self, mask, per_head, records):
        with stage(records, "metadata"):
            metadata = mask_to_pages(mask, self.length, self.group if per_head else 1)
        return self.run_pages(metadata, per_head, records)

    def run_pages(self, metadata, per_head, records):
        indptr, ids, last = metadata
        dense = self.method == "flashinfer"
        page_size = 16 if self.method in ("flashinfer", "quest") else 1
        qo_heads = self.heads if dense else 1 if per_head else self.group
        kv_heads = self.kv_heads if dense else 1
        with stage(records, "plan"):
            self.wrapper.plan(indptr, ids, last, num_qo_heads=qo_heads,
                              num_kv_heads=kv_heads, head_dim=self.dim, page_size=page_size,
                              pos_encoding_mode="NONE", q_data_type=self.q.dtype,
                              kv_data_type=self.k.dtype, sm_scale=1 / math.sqrt(self.dim))
        with stage(records, "attention"):
            query = self.q.view(-1, qo_heads, self.dim)
            self.wrapper.run(query, self.pages, out=self.output.view_as(query))
        return self.output

    def run(self, records=None, keep=False):
        with stage(records, "cache_update"):
            if self.method == "fa2":
                self.fa_k[:, -1].copy_(self.new_k)
                self.fa_v[:, -1].copy_(self.new_v)
            elif self.method == "flashinfer":
                self.fi_k[:, -1].copy_(self.new_k)
                self.fi_v[:, -1].copy_(self.new_v)
            else:
                self.k[:, :, -1].copy_(self.new_k)
                self.v[:, :, -1].copy_(self.new_v)
        if self.method == "fa2":
            with stage(records, "attention"):
                return self.fa2(self.q[:, None], self.fa_k, self.fa_v, causal=True).squeeze(1)
        if self.method == "flashinfer":
            # Metadata/plan is deliberately included even for the fixed-length
            # dense replay, under the same serialized policy as sparse FI paths.
            with stage(records, "metadata"):
                self.dense_metadata[2].fill_(16)
            return self.run_pages(self.dense_metadata, False, records)
        with stage(records, "selector_update"):
            tail = self.k[:, :, -16:]
            self.minimum[:, :, -1] = tail.amin(-2)
            self.maximum[:, :, -1] = tail.amax(-2)
        with stage(records, "selector"):
            candidate = quest_candidates(self.q, self.minimum, self.maximum, self.length, self.budget, pages_only=self.method == "quest")
        if self.method == "quest":
            if keep:
                mask = torch.zeros(self.batch, self.heads, self.length, device=self.q.device, dtype=torch.bool)
                tokens = (candidate.long()[..., None] * 16 + torch.arange(16, device=self.q.device)).flatten(-2)
                mask.scatter_(-1, tokens, True)
                self.last = {"candidate": mask, "final": mask}
            with stage(records, "metadata"):
                requests, width = self.batch * self.heads, candidate.shape[-1]
                offsets = torch.arange(requests, device=self.q.device, dtype=torch.int32) // self.group * (self.length // 16)
                metadata = (torch.arange(requests + 1, device=self.q.device, dtype=torch.int32) * width,
                            (candidate.view(requests, width) + offsets[:, None]).flatten(),
                            torch.full((requests,), 16, device=self.q.device, dtype=torch.int32))
            return self.run_pages(metadata, True, records)
        with stage(records, "estimator_update"):
            self.khat[:, :, -1] = self.quant(self.new_k, 4)
        with stage(records, "estimator"):
            # Preserve the reference's single-query-row GEMV accumulation.
            # A four-query GEMM changes FP16 rounding at top-p boundaries.
            query = self.q.view(self.batch, self.kv_heads, self.group, self.dim)
            scores = torch.empty(self.batch, self.kv_heads, self.group, self.length, device=self.q.device, dtype=self.q.dtype)
            for local_head in range(self.group):
                scores[:, :, local_head:local_head + 1] = torch.matmul(query[:, :, local_head:local_head + 1], self.khat.transpose(-1, -2)) / math.sqrt(self.dim)
            scores = scores.view(self.batch, self.heads, 1, self.length)
            scores.masked_fill_(~candidate.unsqueeze(2), torch.finfo(scores.dtype).min)
        with stage(records, "head_top_p"):
            selected = self.top_p(scores, 0.85) & candidate.unsqueeze(2)
        if self.method == "quest_twi":
            with stage(records, "union"):
                union = selected.view(self.batch, self.kv_heads, self.group, self.length).any(2)
            if keep:
                self.last = {"candidate": candidate, "selected": selected.squeeze(2), "final": union.repeat_interleave(self.group, 1)}
            return self.plan_and_run(union, False, records)
        with stage(records, "group_refinement"):
            shared = self.shared_fn(scores, selected, self.kv_heads, 0.80, "group_max").contiguous()
            final = self.exception_fn(scores, selected, shared, 16).contiguous()
        with stage(records, "compaction"):
            width = max(1, int(shared[:, ::self.group, 0].sum(-1).max().item()))
            group_ids, exception_ids, _, _ = self.module.compact_group_exception_masks(shared, final, self.kv_heads, width, 16)
        if keep:
            self.last = {"candidate": candidate, "selected": selected.squeeze(2), "shared": shared.squeeze(2), "final": final.squeeze(2)}
        # A fixed capacity rule, never a per-sample choice of the faster backend.
        # The current fused kernel's 96-KiB shared-memory limit is authoritative.
        required_smem = self.group * (width + 16) * 4 + 256 * 4
        if required_smem > 96 * 1024:
            self.dispatch = "gas_fi_capacity_fallback"
            return self.plan_and_run(final.squeeze(2), True, records)
        self.dispatch = "gas_fused_group_exception"
        with stage(records, "attention"):
            return self.module.batched_gqa_group_exception_decode_out(self.q, self.k, self.v, group_ids, exception_ids, self.output)


def reference_support(path):
    """Independent quality-code selector/pruner on the first logical sequence."""
    from twilight.pyimpl.quest import quest_selector

    q, k = path.q[:1], path.k[:1]
    repeated = k.repeat_interleave(path.group, dim=1)
    candidate = quest_selector(SimpleNamespace(num_heads=path.heads), q.unsqueeze(2), repeated,
                               chunk_size=16, token_budget=path.budget)
    result = {"candidate": candidate.squeeze(2)}
    if path.method == "quest":
        result["final"] = result["candidate"]
        return result
    scores = torch.matmul(q.unsqueeze(2), path.quant(repeated, 4).transpose(-1, -2)) / math.sqrt(path.dim)
    scores.masked_fill_(~candidate, torch.finfo(scores.dtype).min)
    selected = path.top_p(scores, 0.85) & candidate
    result["selected"] = selected.squeeze(2)
    if path.method == "quest_twi":
        result["final"] = selected.view(1, path.kv_heads, path.group, path.length).any(2).repeat_interleave(path.group, 1)
    else:
        shared = path.shared_fn(scores, selected, path.kv_heads, 0.80, "group_max")
        result["shared"] = shared.squeeze(2)
        result["final"] = path.exception_fn(scores, selected, shared, 16).squeeze(2)
    return result


def support_statistics(path):
    if not path.last:
        return {"candidate_tokens_mean": path.length, "final_loaded_kv_tokens_mean": path.kv_heads * path.length}
    result = {}
    for name, mask in path.last.items():
        result[name + "_per_head_mean"] = float(mask.sum(-1).float().mean())
        union = mask.view(path.batch, path.kv_heads, path.group, path.length).any(2)
        result[name + "_loaded_kv_tokens_mean"] = float(union.sum((-1, -2)).float().mean())
    if "shared" in path.last:
        exceptions = path.last["final"] & ~path.last["shared"]
        result["exception_per_head_mean"] = float(exceptions.sum(-1).float().mean())
    return result
