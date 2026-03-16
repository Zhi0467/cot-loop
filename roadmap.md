# Roadmap - CoT Loop Detection

Last updated: 2026-03-16 14:22 UTC

Scope:
- Build and validate a probe pipeline for CoT loop detection across prefill and completion feature views.
- Quantify generalization across natural eval splits and OOD tasks.

## Current Status
- Milestone 1 gate: complete.
- Milestone 2 gate: complete.
- Milestone 3 gate: complete.
- Active milestone: Milestone 4 (cross-dataset validation).
- Latest result: the common-policy rollout-statistics bundle has now been refreshed under one shared decode policy (`temperature=0.2`, `num_generations=10`, `max prompts <= 800` where applicable) across `MATH-500`, `AIME`, `GPQA`, `MMLU-Pro`, and capped `LiveCodeBench release_v6`. The repaired MC rows now report `GPQA = 34.49%` and `MMLU-Pro = 65.2%` rollout success instead of the stale pre-refresh rows, and the regenerated cross-dataset PDF is published in `outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`.
- Remaining caveat: the original `LiveCodeBench` job crashed after grading and before writing its final JSON, and replay-based repair did not reproduce the stored checkpoint exactly enough to recover `avg_first_loop_prefix_length`. That one metric remains `null` in the recovered capped bundle; a fresh rerun would be required if exact prefix-length telemetry is still needed.
- Active review surfaces:
  - Upstream PR #4 (`Zhi0467/cot-loop`, branch `task/1773451376-rollout-stats`) remains `OPEN` / `DRAFT` at head `d0796c3`, but it no longer covers the current local branch state.
  - Upstream PR #6 (`Zhi0467/cot-loop`, branch `task/1773451376-common-policy-refresh`) is now the live review surface for the refreshed common-policy bundle and the report-generator LaTeX fix.

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

## Milestone 5 - Deployment readiness
Status: future (set 2026-03-13 13:05 UTC)
Success criteria:
- Recommend a default feature view and probe configuration for routine use.
- Document performance tradeoffs, expected failure modes, and cost profile.
