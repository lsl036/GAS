# Fig. 4(a) additional baselines

This is an isolated, reduced-sample follow-up, not a replacement for the original
Fig. 4(a) results. Do not change the frozen original output directories.

## Protocol

- Model: LLaMA-3.1-8B-Instruct, BF16, single GPU, 32K RULER.
- Tasks: niah_single_1, niah_multikey_3, niah_multivalue, vt and cwe.
- Reuse exact prompts, references and GAS metrics from `fig4a_llama31`.
- Default calibration: first 3 samples per task from the existing calibration
  split; formal: first 10 per task from the independent formal split.
- Compare GAS on those exact subsets, not the full-sample GAS summary. Task
  scores are substring-match scores, equally macro-averaged over the five tasks.
- The candidate grid is LessIsMore's total shared-token budget 128 and 512,
  with recent window ratio 0.25, selection layers 2 and 13. The frozen choice
  uses calibration only: highest score if KV is in [100%, 105%] of GAS,
  otherwise nearest above, otherwise nearest. This is an extremely small pilot,
  not statistically strong enough for a definitive paper claim.
- Results append per sample and can be resumed by rerunning the identical
  command. The manifest rejects changes to protocol in an existing output dir.
  If a process is interrupted while writing JSON, inspect the final line before
  resuming. Neither existing GAS data nor an existing incompatible run is removed.

## LessIsMore fidelity and accounting

The included upstream checkout is `DerrickYLJ/LessIsMore` at
`fa3d23a45fde680ac220394e7c569f5e98ca144e`. The adapter uses its actual LLaMA
`modify_llama_lim.py` implementation. Its upstream launcher passes `lim` to a
helper that installs the forward only for `tidal`; the adapter calls that helper
with the latter sentinel, without changing its scoring. It also provides a
typing-only `LossKwargs` compatibility shim for the local Transformers version.

`loaded_kv` reports mean logical selected KV tokens summed over the 8 KV groups,
averaged over decode steps and layers 2-31, consistent with our Fig. 4 metric.
The original implementation computes full attention at selection layers 2 and
13; those layers must be counted as dense. At 32K the lower bound from these two
layers alone is approximately `2/30 * 8 * 32768 = 17476`, before the other
layers are included. A 128-budget, 32619-token smoke test reported 18353.33,
while the sparse-only layers reported 1024. Accordingly this method cannot be
called budget-matched with GAS (~1297 on the old calibration aggregate) in this
protocol. `sparse_only_kv` is a diagnostic only and must not be substituted for
`loaded_kv` in the primary plot. Actual cache remains dense; upstream PyTorch
code computes dense logits and masks them, so neither metric proves physical
memory savings or runtime speedup.

Outputs: `selection.json` (calibration, including matching status),
`formal_comparison.csv` (per-task and macro), and per-sample JSONL files.

## RULER and LongBench 13-Task Pilot

`run_quality_matrix.sh` runs K=128 and K=512 with 10 samples per task. It uses
the existing RULER synthetic dataset at 16K, 32K and 64K and the first ten
LongBench records verified against the existing E5 sample manifest. Existing
Fig. 4(a) predictions are not reused because they use different dataset splits.
The output is `benchmark/results_quality/lessismore_13task_10/`. Predictions
are checkpointed per sample; rerun the same command to resume. An OOM stops the
current task, continues other tasks, and leaves the affected macro score blank.
Compare against baselines recomputed on these same ten samples, not their
original 50-sample aggregate. This accuracy pilot is not an efficiency test.

`run_rate_quality.sh` is the separate dynamic-rate variant. It uses the same
tasks and samples, with `K=max(1, int(current_kv_length * rate))` recomputed at
each decode step for rates 4% and 8%; it does not use the nominal context length
as a fixed K. It writes `benchmark/results_quality/lessismore_rate_13task_10/`
and records per-sample logical KV and minimum/maximum K. The all-layer KV
metric includes the two full-attention selection layers and is not a hardware
memory-traffic measurement. Matching the nominal rate is not matching the
loaded KV budget across different selectors.

## HShare status

The official checkout `wuhuaijin/HShare`, commit
`523bea0cbd26f7d987f6609b77c6f58387f7dbb1`, is present for auditing but is not
yet a runnable 32K LLaMA-3.1 baseline. Its evaluation LLaMA implementation
materializes dense `[batch, 32, length, length]` attention and a second grouped
score matrix during prefill; one BF16 matrix at 32K alone is 64 GiB, exceeding
the A100 40 GB before model/cache memory. It also uses older Transformers cache
and RoPE APIs and does not include a LLaMA-3.1-8B channel configuration.
Running a shorter context, substituting another model's calibration, silently
disabling sharing, or relabeling DS as HShare would not reproduce this comparison.
Adapting it requires validating an exact memory-efficient prefill and sharing
configuration against the official implementation at short length, preparing
model-specific channels, and only then testing 32K. No HShare result is claimed.
