# Prompt-Profile Risk-Screen Decision

Last updated: 2026-03-30 03:10 UTC

## Question

- After the rebuilt `majority_s_0.5` control table, the remaining open question was no longer "does prompt length ever work?" It was: under the fixed prompt-profile setup, which prompt-prefill score actually isolates the most degenerate prompts across datasets once the continuous metadata controls are trained for real?

## Artifacts

- Local analysis bundle: `../outputs/prompt_profile_risk_controls_20260330/`
- Cross-dataset summary JSON: `../outputs/prompt_profile_risk_controls_20260330/cross_dataset_summary.json`
- Cross-dataset bucket table: `../outputs/prompt_profile_risk_controls_20260330/cross_dataset_bucket_summary.csv`
- Metadata-baseline metric table: `../outputs/prompt_profile_risk_controls_20260330/metadata_baseline_metrics.csv`

## What Was Run

- Finished bundles: `GPQA`, `AIME` seed-`0`, `MATH-500`, `MMLU-Pro`, `LiveCodeBench`
- Learned scores compared on the saved test prompts:
  - `majority_s_0.5`
  - `p_loop = E_r[1{rollout r loops}]`
  - `mean_relative_length = E_r[L_r / E]`
- First real trained metadata-only controls for the continuous heads:
  - `p_loop`: logistic metadata model on prompt-disjoint train prompts, using `prompt_length`, `effective_budget`, or both
  - `mean_relative_length`: linear metadata model on the same train prompts and the same feature sets
- Operational test:
  - rank test prompts by predicted score
  - keep the top predicted-risk `20%`
  - measure actual loop rate, actual max-length-hit rate, and actual accuracy when available

## Main Result

- `p_loop` is now the best-supported main training objective for degeneracy screening.
- On the top-risk-20% bucket test, the selected `p_loop` variant wins loop-rate enrichment on `AIME`, `MATH-500`, `MMLU-Pro`, and `LiveCodeBench`, and it roughly ties `GPQA` with `majority_s_0.5`.
- On the four datasets where prompt-level accuracy is available (`GPQA`, `AIME`, `MATH-500`, `MMLU-Pro`), the `p_loop` bucket is also the lowest-accuracy bucket every time.
- Averaged across datasets, loop-rate enrichment is about:
  - `2.86x` for `p_loop`
  - `1.87x` for `majority_s_0.5`
  - `1.77x` for `mean_relative_length`
  - `1.24x` for the strongest metadata baseline
- `mean_relative_length` still ranks its own target well and remains useful as a secondary utility / budget-consumption score, but it is weaker than `p_loop` as the main bad-prompt screen.
- `majority_s_0.5` still has real signal, especially for cap-prone or geometry-heavy prompts, but it is no longer the best-supported cross-dataset screen for degeneracy.

## Control Read

- The continuous metadata gap is now closed for the current saved bundles.
- The strongest metadata baseline is almost always a one-feature model, not the joint `prompt_length + effective_budget` control.
- In practice, the joint baseline did not produce a new winner on any dataset.
- The metadata controls remain weaker than the selected `p_loop` screen on the operational bucket test.

## Decision

- Main screening objective: `p_loop`
- Secondary utility head when a budget/proxy score is still wanted: `mean_relative_length`
- Keep `majority_s_0.5` as a control / auxiliary cheap screen, not the headline target
- Do not reopen direct `p_cap` now; only reopen it if a later slice shows cap-hit isolation matters operationally beyond what `p_loop` already surfaces

## Caveats

- This is a retrospective comparison on finished saved bundles, not a fresh prospective deployment run.
- `LiveCodeBench` still lacks prompt-level accuracy in the recovered projection artifact, so the low-accuracy comparison is only `4 / 5` datasets.
- `GPQA` and especially `AIME` are still small enough that a few prompts can move the bucket numbers.
- The bucket result is reported at one operating point (`top 20%`); it is strong evidence for target choice, not the full operating-characteristic curve.
- Lower bucket accuracy supports the degeneracy read, but some of that signal can still mix prompt difficulty with looping rather than separating them cleanly.
