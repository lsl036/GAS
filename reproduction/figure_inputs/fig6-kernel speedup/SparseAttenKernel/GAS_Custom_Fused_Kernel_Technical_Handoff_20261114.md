# GAS-Custom Fused Sparse Attention: Technical Handoff

**Status:** research prototype and controlled attention replay, not an integrated serving implementation.  
**Purpose:** provide code-grounded technical details and measured results for paper drafting.  
**Date:** 2026-11-14.

## 1. What Is Implemented

GAS-custom is a CUDA sparse decode-attention prototype designed for a GQA group with four query heads sharing one KV head. It consumes a compact **group-shared token support plus per-head exceptions**. Its main CUDA kernel combines sparse K/V access, QK dot products, online softmax and weighted V accumulation. A separate CUDA kernel merges partition-level softmax statistics and partial outputs.

This is **not** a single kernel spanning Quest selection, Twilight estimator, top-p, GAS refinement and attention. The refinement, compact support preparation and attention components are separate. The experiments below do not establish an integrated end-to-end inference speedup.

Primary code:

- Attention prototype: [`gas_cooperative_shared_pool.cu`](../../Twilight_Ours/csrc/src/gas_cooperative_shared_pool.cu), particularly `group_attention_body`, `group_attention`, `merge_parts`, `launch` and `decode_impl`.
- Python semantics of shared selection and exceptions: [`gqa_refine.py`](../../Twilight_Ours/twilight/pyimpl/gqa_refine.py).
- Experimental CUDA group refinement: [`gas_compact_refine.cu`](../../Twilight_Ours/csrc/src/gas_compact_refine.cu), particularly `prepare_group_scores` and `group_tagged_normalized_prepared`.
- Experimental Triton exception selection: [`gas_compact_refinement.py`](../../Twilight_Ours/benchmark/efficiency/gas_compact_refinement.py), `exception_stage_unified`.
- Matched Qasper replay: [`bench_qasper_gas_paired.py`](../../Twilight_Ours/benchmark/efficiency/bench_qasper_gas_paired.py).

## 2. Inputs And Semantics

For request `b`, KV group `g` and query head `h` within that group:

```text
shared_ids[b, g, :]     = S_g              # token IDs shared by all heads in the group
exception_ids[b, h, :]  = E_h              # head-specific token IDs
shared_count[b, g]      = |S_g|
exception_count[b, h]   = |E_h|
attention support[h]    = S_g union E_h
```

In the tested GAS setting, `S_g` is obtained from group-max scoring followed by group top-p `p_g=0.80`; `E_h` contains up to 16 selected head-specific residual tokens outside `S_g`. The candidate support supplied to this refinement is the output of Quest plus Twilight top-p, not the entire context. Refinement is not timed in the author-timer attention-only table below.

The attention kernel trusts the passed prefix counts, unique indices and disjoint exceptions; it does not itself validate those metadata invariants. For the measured prototype, Q is `[batch, 32, 128]`, K/V use a token-major shared-pool layout `[pool_tokens, 8, 128]`, and inputs/output are FP16. The source also dispatches BF16, but the Qasper figures reported here use FP16 after converting captured BF16 Q/K/V. Head dimension is fixed at 128 and the kernel checks a 4:1 query/KV head ratio. These constraints must not be described as generic support for all GQA layouts.

## 3. Attention Kernel Techniques

### 3.1 Group-cooperative shared KV access

Work is indexed by `(batch, KV group, partition)`. Each group consists of four query heads. For a token in `S_g`, a thread segment loads the token's K and V vector slices once and reuses them when updating all four heads' QK/softmax/PV accumulators. A head-specific exception is processed only for its owning head.

The implementation uses 16-lane segments, vectorized 16-byte (`uint4`) Q/K/V accesses and warp shuffle reductions for the QK dot product. It scales logits by `1/sqrt(128)`. These code-level reuse properties do **not** prove a specific reduction in HBM transactions: actual memory traffic was not measured with a hardware profiler.

### 3.2 Online softmax and PV accumulation

For each head the kernel maintains a running maximum `m`, denominator `l` and value accumulator `a` in FP32. For a new logit `x` and value `v`, it updates the state with rescaling:

```text
m_new = max(m, x)
l_new = l * exp(m - m_new) + exp(x - m_new)
a_new = a * exp(m - m_new) + exp(x - m_new) * v
```

This avoids materializing a dense attention-logit or probability tensor and keeps the softmax numerically stable. FP16 Q/K/V values are converted for FP32 accumulation, and the final output is cast back to the input dtype. Shared and exception tokens participate in the same head-specific online softmax; exception scores are not normalized separately.

### 3.3 Segment and partition parallelism

Within a block, multiple 16-lane segments process token subsets. Their partial outputs, maxima and denominators are combined through shared memory. Longer selected supports are also split across partitions, with a second `merge_parts` kernel combining partition statistics using stable softmax rescaling.

The measured `coop8p4` call uses `segments=8`, at most `4` partitions and `chunk=2`. The effective partition count is computed from the group support and per-head exception counts. The standard `decode` path launches the attention kernel and the merge kernel; `launch` also allocates an FP32 partial buffer. The presence of the merge kernel matters when describing the implementation as “fused”: QK/online-softmax/PV are fused **within the main attention kernel**, not necessarily into one CUDA launch.

There is a separate `decode_compact` work-queue variant in the same source, but it is **not** the `coop8p4` standard `decode` path used for the paired Qasper table. Do not attribute its queue behavior to the reported measurements.

## 4. Refinement Optimizations And Boundary

These optimizations belong to the separate experimental refinement path, not to the fused attention kernel itself:

1. `prepare_group_scores` processes per-head candidate scores before the group selection kernel; `group_tagged_normalized_prepared` normalizes group scores once rather than repeatedly dividing during threshold search. Both CUDA launches are part of this refinement path.
2. Tagged epoch flags track membership without clearing a full dense flag buffer each time. The prototype uses mutable per-group epoch state; concurrency and epoch rollover require production treatment.
3. The unified Triton exception path reduces separate per-head dispatches and selects residual exceptions outside the shared support.
4. Compact IDs and counts feed the custom attention prototype. In the two-request test, local IDs are shifted to offsets in a flattened two-pool KV layout. Pre-shifted IDs are used for the attention-only timing; the custom `refinement + attention` replay includes the offset operations.

The author-timer table in Section 5 excludes all refinement and metadata construction. A separate CUDA Graph replay includes refinement for GAS-custom; its FI comparison uses a **static preplanned mask**, so it does not implement dynamic refinement-to-FI metadata creation or re-planning. Its times cannot establish the relative performance of fully connected dynamic pipelines.

## 5. Paired Qasper Attention Results

### Conditions

- A100 40GB; 16K and repeated-context 32K Qasper traces; samples 15 and 54 alternated in the batch using two physical KV pools.
- 28 non-search layers: 3–12 and 14–31, one teacher-forced question token per trace.
- Same captured GAS mask for FI head, FI group and GAS-custom; Quest 8% candidate, Twilight top-p 0.85, GAS group-max top-p 0.80 plus exception16.
- FI head: paged decode adapter. FI group: paged prefill adapter with exact per-head mask. These are FlashInfer adapters, not native Quest/Twilight/GAS kernels.
- Author helper `torch.utils.benchmark`: 5 warmups, 50 calls per round, 3 randomized-interleaved rounds; median of round means for each layer, then arithmetic mean across 28 layers. Units: μs per layer.
- This measures attention-call latency including host/framework dispatch. It excludes selector, estimator, both top-p stages, refinement, index generation, FI planning and model execution. It is not model TPOT or GPU-kernel-only latency.

| Context | Batch | FI head | FI group | GAS-custom | FI group / custom |
|---|---:|---:|---:|---:|---:|
| 16K | 32 | 98.29 | 119.44 | 71.08 | 1.68x |
| 16K | 64 | 96.46 | 116.71 | 72.07 | 1.62x |
| 16K | 96 | 97.86 | 117.79 | 81.19 | 1.45x |
| 16K | 128 | 104.95 | 125.86 | 97.09 | 1.30x |
| 32K | 32 | 99.35 | 118.68 | 75.47 | 1.57x |
| 32K | 64 | 101.88 | 120.21 | 82.64 | 1.45x |
| 32K | 96 | 106.88 | 122.54 | 99.89 | 1.23x |
| 32K | 128 | 112.77 | 126.35 | 123.21 | 1.03x |

The custom attention call is faster than FI group in these eight author-timer configurations, and faster than FI head in seven; at 32K/B128 it is slower than FI head. These are **not** equal-implementation comparisons: the custom kernel and FI adapters have different GPU and host-side implementations.

### Important counter-result: CUDA Graph event replay

The earlier matched-mask experiment used CUDA Graph events, a distinct timing regime that largely removes ordinary per-call host dispatch from replay. In that measurement custom attention was **slower** than grouped FI in all eight configurations:

| Context | Batch | FI group, μs | GAS-custom, μs | Custom / FI |
|---|---:|---:|---:|---:|
| 16K | 32 | 31.39 | 39.22 | 1.25x |
| 16K | 64 | 42.08 | 57.27 | 1.36x |
| 16K | 96 | 54.03 | 75.82 | 1.40x |
| 16K | 128 | 66.14 | 94.44 | 1.43x |
| 32K | 32 | 39.07 | 51.55 | 1.32x |
| 32K | 64 | 50.40 | 73.66 | 1.46x |
| 32K | 96 | 64.04 | 96.64 | 1.51x |
| 32K | 128 | 77.87 | 121.33 | 1.56x |

The two tables differ in timing methodology and must not be combined as if they measured the same thing. The observed reversal is consistent with the impact of host/framework dispatch and backend behavior, but the individual contribution of each overhead was not separately measured. Neither table establishes model end-to-end speedup.

## 6. Correctness Evidence

For every one of the 224 `(context, batch, layer)` configurations:

- The reconstructed GAS masks for both unique requests matched the captured GAS masks exactly; post-timing refinement masks were also checked.
- FI head, FI group and GAS-custom outputs were checked over the whole repeated batch. Maximum absolute FI-group/custom and FI-head/group errors were both `0.0009765625`.
- Custom outputs for the two unique requests were checked against an FP32 masked-attention reference with `atol=0.02, rtol=0.02`; the largest absolute error was `0.0014172`.

These are replay correctness checks on two captured requests and 28 layers, not proof of general numerical equivalence or model-level accuracy for the new custom kernel. The reconstructed BF16 estimator scores are derived from captured Q/K, not saved estimator logits, and the resulting masks were checked against saved masks.

## 7. Writing Guidance

Supported, scoped wording:

> We prototyped a group-cooperative sparse decode attention implementation that reuses each shared KV token across the four query heads of a GQA group, accumulates head-specific exceptions, and combines QK, online softmax, and value accumulation in its main CUDA kernel. On a two-request Qasper trace replay, measured attention-call latency varied with the backend and timing methodology; the CUDA Graph replay did not show a GPU-time improvement over the grouped FlashInfer adapter.

Avoid claiming any of the following from these measurements:

- A verified end-to-end model or serving TPOT speedup.
- A guaranteed reduction in resident KV-cache capacity, or measured HBM traffic reduction.
- A native official Twilight or Quest kernel comparison in this paired experiment.
- A single-kernel implementation of selection, refinement, attention and partition merge.
- A consistent GPU-kernel speedup over FlashInfer; the CUDA Graph result contradicts that claim for the measured inputs.
- Generality beyond the tested 4:1 GQA, head dimension 128, two-request shared-pool workload, FP16 replay and two sampled prompts.

The synthetic budget exploration in [`Synthetic_Union_Budget_Exploration_20261114.md`](Synthetic_Union_Budget_Exploration_20261114.md) is a post-selected, deliberately skewed stress test with unrelated synthetic estimator logits and Q/K/V. It must not be presented as representative Qasper/RULER performance or as a matched-quality result.

## 8. Artifacts

- Author-timer [summary](../../Twilight_Ours/benchmark/results_efficiency/qasper_gas_author_20261114/summary.md) and [CSV](../../Twilight_Ours/benchmark/results_efficiency/qasper_gas_author_20261114/summary.csv); per-layer JSON files and per-round means are in each `16k_b*` / `32k_b*` directory.
- CUDA Graph [summary](../../Twilight_Ours/benchmark/results_efficiency/qasper_gas_paired_20261114/summary.md) and [CSV](../../Twilight_Ours/benchmark/results_efficiency/qasper_gas_paired_20261114/summary.csv).
- [Author-timer summarizer](../../Twilight_Ours/benchmark/efficiency/summarize_qasper_gas_author.py) and [CUDA Graph summarizer](../../Twilight_Ours/benchmark/efficiency/summarize_qasper_gas_paired.py).
- Preserve the per-run `config.json`, original trace manifests, per-layer JSON, source code and environment details together when transferring results for external analysis.
