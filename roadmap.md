# Roadmap - CoT Loop Detection

Last updated: 2026-03-15 12:15 UTC

Scope:
- Build and validate a probe pipeline for CoT loop detection across prefill and completion feature views.
- Quantify generalization across natural eval splits and OOD tasks.

## Current Status
- Milestone 1 gate: complete.
- Milestone 2 gate: complete.
- Milestone 3 gate: complete.
- Active milestone: Milestone 4 (cross-dataset validation).
- Latest result: under the repaired rollout-statistics v2 contract, `MATH-500`, `AIME`, `GPQA`, and capped `MMLU-Pro` are complete and reported. The only remaining blocker is the final capped `LiveCodeBench release_v6` leg, which was still running cleanly at `402 / 800` prompts on the pinned seven-GPU path at `2026-03-15 12:15 UTC`.
- Active experiment: the capped `LiveCodeBench` finish watch on Slurm job `1330`. The authoritative scratch stats dir still has no `livecodebench_release_v6__test__Qwen_Qwen3-1.7B.json` or paired `__lcb_records.json`, so the project remains in monitor/report mode rather than a new training or repair round.
- Active review surfaces:
  - Upstream PR #4 (`Zhi0467/cot-loop`, branch `task/1773451376-rollout-stats`) remains `OPEN` / `DRAFT` at head `d0796c3`, but it no longer covers the current local branch state.
  - The local follow-up branch already carries the stale-artifact fix (`d941986`) plus the post-rerun doc state; bounded `review_project.sh --base main` still does not reach a terminal verdict here, so the branch is not locally review-cleared yet.

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
