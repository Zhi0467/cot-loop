# Roadmap - CoT Loop Detection

Last updated: 2026-03-19 23:05 UTC

Scope:
- Build and validate a probe pipeline for CoT loop detection across prefill and completion feature views.
- Quantify generalization across natural eval splits and OOD tasks.

## Current Status
- Milestone 1 gate: complete.
- Milestone 2 gate: complete.
- Milestone 3 gate: complete.
- Active milestone: Milestone 4 (cross-dataset validation).
- Latest result: the common-policy rollout-statistics bundle remains refreshed under one shared decode policy (`temperature=0.2`, `num_generations=10`, `max prompts <= 800` where applicable) across `MATH-500`, `AIME`, `GPQA`, `MMLU-Pro`, and capped `LiveCodeBench release_v6`. The live objective remains an in-distribution prompt-level rollout-profile predictor around a cap-robust normalized-length tail target plus dense normalized-length regression, with `p(max_length_hit)` kept as an operational eval or diagnostic head. A 2026-03-19 Athena cross-check sharpened the fallback choice: if only one head ships first, use `s_0.9 = P(L / E >= 0.9)` rather than `mu_log_rel`, keep correctness out of the first loss, and control the first GPQA pass so `E` is nearly constant enough that the target does not reduce to prompt length. That single-head path is now implemented in the repo for prefill views: repeated rollouts can be aggregated into prompt-level `s_t` soft targets, the builder now supports task-aware `GPQA` / `MMLU-Pro` prompt formatting, and the canonical SLURM launcher exposes the needed model/task/decode knobs. The first remote GPQA prompt-profile pilot was staged on the corrected PR #7 head with a local `gpqa_diamond.csv` mirror, `temperature=0.2`, `num_generations=10`, `max_tokens=30000`, and per-layer ensemble scoring, but it was canceled rather than left pending because `wth-gpu-01` is currently fully allocated to another user's 8-GPU job through the scheduled end time `2026-03-21 20:18 UTC`.
- Remaining caveat: the original `LiveCodeBench` job crashed after grading and before writing its final JSON, and replay-based repair did not reproduce the stored checkpoint exactly enough to recover `avg_first_loop_prefix_length`. That one metric remains `null` in the recovered capped bundle; a fresh rerun would be required if exact prefix-length telemetry is still needed.
- Immediate implementation caveat: this workspace still did not have a local Torch runtime / project virtualenv, so the code path was syntax-checked locally and the first real Torch-backed build/train execution still needs the remote pilot window to open.
- Active review surfaces:
  - Upstream PR #7 (`Zhi0467/cot-loop`, branch `task/1773870804-prompt-profile-probe`) is the live review surface for the prompt-profile implementation at head `48b81cb`, with the probability-target fixes, GPQA prompt-path fix, and launcher updates all published together.
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

## Milestone 5 - Deployment readiness
Status: future (set 2026-03-13 13:05 UTC)
Success criteria:
- Recommend a default feature view and probe configuration for routine use.
- Document performance tradeoffs, expected failure modes, and cost profile.
