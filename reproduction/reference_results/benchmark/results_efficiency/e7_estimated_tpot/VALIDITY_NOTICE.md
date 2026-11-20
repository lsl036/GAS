# E7 audit notice: aggregate TPOT claims withdrawn pending repair

Date: 2026-10-23. Raw files are retained for audit, not overwritten.

The current E7 summary does **not** establish a reliable GAS/Quest or
GAS/Quest+Twilight Estimated TPOT speedup. The issue is the benchmark
implementation, not the user's launch parameters or the model quality results.

Source audit found:

1. `selection_callable()` times operations on random scores, while attention
   consumes independently generated, profile-sized supports. Selection outputs
   do not feed attention. Actual Quest page scoring and the 4-bit estimator are
   absent. GAS aggregation operates on per-head top-k positions without aligning
   their token IDs; this is not the group-max algorithm used in quality tests.
2. Sparse methods multiply their own sparse attention latency by both the two
   retained dense layers and the 30 sparse layers. The two dense layers must use
   a common measured dense backend instead.
3. Common work uses random shape-matched matrices, not checkpoint weights, and
   does not implement all required RoPE/cache update, norm/residual, or output
   sampling operations. Repeating one component estimate under multiple trace
   IDs is not independent trace evidence.
4. Quest's final attention uses our simple one-block-per-query-head adapter.
   Its poor scaling at 8192 candidates does not measure the official Quest
   implementation or an optimized compatible backend. Budget remains 8192;
   changing it to 405 would change the comparison.
5. Dense OOM/unavailable points were measured with unnecessary simultaneous
   buffers. They are limitations of this harness, not demonstrated capacity
   limits of FA2 or FlashInfer.

Kernel correctness checks validate frozen-support attention only, not the
connected selection/refinement pipeline. Previously quoted 8x--12x vs Quest
and 2.07x--3.00x vs Quest+Twilight must not be used as paper conclusions.
Model quality and logical loaded-KV statistics are separate evidence; this
notice does not invalidate those experiments.

## Immediate repair and interpretation

`benchmark/efficiency/bench_quest_budget_sweep.py` v2 compares the legacy kernel
with FlashInfer using exactly the same Q/K/V, 8192-token supports, and FP32
reference. Each query head becomes a logical request into the original KV
storage. No full KV expansion/gather is needed for attention. Gather is a
separate, preallocated diagnostic. Results are saved incrementally; kernel
counts can be measured with `--profile-kernels` rather than inferred.

This audit is **kernel-only**, with static supports and plan outside timing.
It supplies an optimized compatible attention control, not corrected TPOT.

## Conditions for new Estimated TPOT results

1. Connect actual Quest selection, low-bit estimation, Twilight pruning, GAS
   refinement, and dynamic compaction to the attention inputs. All sparse
   variants start from the same upstream candidate definition and Q/K/V.
2. Benchmark efficient compatible attention paths for Quest and Twilight as
   well as GAS; include required dynamic metadata/plan with consistent timing
   boundaries. Do not add a separate gather to a kernel that never uses it.
3. Account for retained dense layers with the same frozen dense backend, and
   measure the complete common/output work required by E7. Use actual traces
   or explicitly identify and validate any synthetic fallback.
4. Re-run all three sparse methods under this corrected protocol. Label any
   component-only result as such. Do not patch only Quest's attention number
   into the old summaries and call the result corrected TPOT.

Please give the paper agent this notice together with the retained E7 files.

## Replacement script

The separately implemented E7 v2 is launched with
`benchmark/efficiency/run_e7_v2.sh` and writes to
`benchmark/results_efficiency/e7_v2_connected/`. It uses actual checkpoint
weights and real decode traces, connects selection outputs to attention,
retains measured FA2 for layers 0/1, and labels GAS capacity fallback explicitly.
Its small validation passed; the full 40-point formal matrix is still to be
run by the user. Do not merge the old E7 totals with the new experiment.

See `thoughts/Efficiency/E7_v2_Connected_TPOT_Runbook.md` for commands, timing
boundaries, verification evidence and expected deliverables.
