# Roadmap - CoT Loop Detection

Last updated: 2026-03-21 23:53 UTC

Scope:
- Build and validate a probe pipeline for CoT loop detection across prefill and completion feature views.
- Quantify generalization across natural eval splits and OOD tasks.

## Current Status
- Milestone 1 gate: complete.
- Milestone 2 gate: complete.
- Milestone 3 gate: complete.
- Active milestone: Milestone 4 (cross-dataset validation).
- Latest result: the common-policy rollout-statistics bundle remains refreshed under one shared decode policy (`temperature=0.2`, `num_generations=10`, `max prompts <= 800` where applicable) across `MATH-500`, `AIME`, `GPQA`, `MMLU-Pro`, and capped `LiveCodeBench release_v6`. The repaired MC rows now report `GPQA = 34.49%` and `MMLU-Pro = 65.2%` rollout success instead of the stale pre-refresh rows, and the regenerated cross-dataset PDF is published in `outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`.
- Prompt-profile direction: the live in-distribution objective is no longer just a pre-pilot choice. The first real GPQA `s_0.9` pilot finished cleanly and exposed a narrower problem: on this `GPQA` / `max_tokens=30000` surface, `s_0.9` collapses exactly to `p(max_length_hit)` on the pilot prompts, so the next main head should shift to a denser terminal statistic (`mean_relative_length` first, or a lower-threshold `s_t`) while still keeping cap-hit/correctness diagnostic-only. A same-archive `mean_relative_length` retrain is already directionally better (`eval_spearman +0.0647` instead of `-0.2799`) but still loses on MSE/MAE to trivial baselines, so the next round needs more prompts as well as the denser target. Prompt-token-count and/or effective budget `E` remain the leakage baselines, and any single-layer probe should still default to the last layer.
- Runtime status: the resumed GPQA prompt-profile pilot (`1477`) is now complete rather than pending. It finished in `37m22s` on GPUs `0-2`, wrote the full prompt-profile archive plus probe checkpoints, and showed that the 3-GPU runtime path is viable. The blocker is now objective usefulness, not Slurm/runtime feasibility.
- Remaining caveat: the original `LiveCodeBench` job crashed after grading and before writing its final JSON, and replay-based repair did not reproduce the stored checkpoint exactly enough to recover `avg_first_loop_prefix_length`. That one metric remains `null` in the recovered capped bundle; a fresh rerun would be required if exact prefix-length telemetry is still needed.
- Immediate implementation caveat: this workspace still did not have a local Torch runtime / project virtualenv, so the code path was syntax-checked locally and the first real Torch-backed build/train execution still needs the remote pilot window to open.
- Active review surfaces:
  - Upstream PR #7 (`Zhi0467/cot-loop`, branch `task/1773870804-prompt-profile-probe`) is the live review surface for the prompt-profile implementation.
  - Upstream PR #6 (`Zhi0467/cot-loop`, branch `task/1773451376-common-policy-refresh`) remains the published review surface for the common-policy rollout bundle.

## Milestone 1 - Pipeline and multi-view infrastructure
Status: done (2026-03-05 18:45 UTC)
Success criteria:
- Dataset builder supports prefill and completion feature views in a single pass.
- Training and eval support explicit feature-key selection, multi-view reuse, and imbalance-aware metrics.
- Slurm launchers exist for shared-dataset and k=5 three-view sweeps.

## Milestone 2 - Completion-vs-prefill baseline package
Status: done (2026-03-05 18:45 UTC)
Success criteria:
- Consolidated findings doc and PDF reflect the k=5 three-view results.
- PR #2 includes the stacked all-layer prefill, metrics, and metadata follow-up code paths.
- Completion-view and prefill-view performance are directly comparable on shared labels.

## Milestone 3 - Metadata-aware prefill residual validation
Status: done (2026-03-13 21:44 UTC)
Success criteria:
- Re-run prefill follow-ups under fixed labels / metadata-aware controls instead of raw composition drift.
- Determine whether any prompt-summary or augmentation beats the metadata-aware last-token anchor on matched evaluation.
- Record the best prefill candidate and its incremental lift versus both metadata-only and anchor-only baselines.

## Milestone 4 - Cross-dataset validation
Status: in progress (set 2026-03-14 02:21 UTC)
Success criteria:
- Replicate the current best prefill/completion findings on additional evaluation sets or model variants.
- Measure whether the preferred feature view survives varied token budgets and source mixes.
- Separate true robustness gains from prompt-length or source-composition effects.
- Establish a useful in-distribution prompt-level predictor objective for fixed model+policy behavior, with metadata-controlled evaluation.
- Make that objective less brittle to the chosen decode ceiling than a pure max-length-hit label.

Activity log:
- 2026-03-20 05:36 UTC: rebased the prompt-profile branch onto the newer upstream `main` state, resolved the docs-layer merge drift, and extended the implementation so repeated-rollout builds can supervise either direct `s_0.9` or `mean_relative_length` while emitting one reusable `diagnostics/prompt_rollout_archive.jsonl` bundle per dataset build.
- 2026-03-21 17:18 UTC: the first real GPQA `s_0.9` pilot finished end to end on the 3-GPU path (`37m22s`) and materially changed the recommendation. The runtime path is fine, but the target is not: on this `32 / 16` pilot, `s_0.9` equals `p(max_length_hit)` for every prompt, the probe's eval Brier (`0.03456`) is slightly worse than a constant baseline (`0.03434`), and eval Spearman is negative (`-0.2799`). The next run should therefore keep the same prefill/data-view design but switch the main head to `mean_relative_length` (or lower the tail threshold) instead of treating `s_0.9` as the settled first objective.
- 2026-03-21 17:24 UTC: reused the finished rollout archive and prefill shards to train `mean_relative_length` on the exact same prompts with no second rollout. This confirms the direction but not success yet: the best regression run improves eval Spearman to `+0.0647`, but its eval MSE/MAE (`0.0503 / 0.1549`) still trail both a constant baseline (`0.0311 / 0.1342`) and a prompt-length-only linear fit (`0.0422 / 0.1425`). The next experiment therefore needs a denser target plus a larger prompt count, not only a label swap on the same 48-prompt slice.
- 2026-03-21 23:53 UTC: generated the first corrected activation-based visualization from the saved GPQA prompt-profile pilot rather than from rollout text. The exporter now joins `manifest.json`, saved prefill shards, prompt-profile JSONLs, and `prompt_rollout_archive.jsonl` into one prompt/rollout projection table; the rendered PCA panels show that correctness is the clearest broad gradient in last-layer prefill space, while `max_length` hits remain visible but fragmented across several prompt islands and loop labels stay broader and more mixed than cap hits.

## Milestone 5 - Deployment readiness
Status: future (set 2026-03-13 13:05 UTC)
Success criteria:
- Recommend a default feature view and probe configuration for routine use.
- Document performance tradeoffs, expected failure modes, and cost profile.
