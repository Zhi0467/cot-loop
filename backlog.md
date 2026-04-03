# CoT Loop Detection Backlog

Last updated: 2026-04-02 23:04 UTC

## Immediate Next Experiments

- Run the next full train pass on the locked target pair: regression `mean_relative_length` plus binary `majority_s_0.5`.
  - Canonical plan note: `docs/prompt-profile-full-train-plan-2026-04-02.md`
  - Canonical PDF: `outputs/prompt_profile_full_train_plan_20260402/prompt_profile_full_train_plan_20260402.pdf`
  - Keep the run contract fixed to the saved prompt-profile surface: `Qwen/Qwen3-1.7B`, `temperature=0.2`, `num_generations=4`, `loop_n=30`, `loop_k=20`, prompt-prefill only.
  - Judge `mean_relative_length` by held-out regression fit plus metadata lift.
  - Judge `majority_s_0.5` by held-out classification quality on the same prompt-disjoint setup.
  - Use ensemble as the main surface and last-layer as the cheap control.
  - Keep `best_loss` as the main checkpoint for target-fit reporting; keep `best_rank` diagnostic only.
- Keep the old bucket test in the diagnostic lane only.
  - The `top 20%` loop-enrichment slice is still useful downstream.
  - It should not re-open target selection by itself.
- Reopen direct `p_loop` only if a later prospective run justifies it as the main train target.
  - Until then, keep it as the loop-specific analysis head rather than the default execution surface.
- Recover `LiveCodeBench` prompt-level accuracy only if the 5/5 accuracy table becomes operationally necessary.
  - The current recovered projection surface still lacks prompt-level correctness, so the completed bucket test is `4 / 5` datasets for accuracy even though the loop / cap comparisons are complete.
- Reopen direct `p_cap` only on contradiction.
  - The finished metadata + bucket pass still does not justify reopening it by default.
  - Only reopen `p_cap` if a later slice shows cap-hit isolation matters beyond what the locked pair and `p_loop` already surface.

## Fixed Experimental Surface

- Keep the predictor input to prompt-prefill activations only.
- Stay in-distribution and prompt-disjoint.
- Keep one selected layer or per-layer independent probes with late aggregation.
- Keep the decode policy fixed at `temperature = 0.2` while settling the objective question.

## Measurement And Reporting Gaps

- The explicit cross-dataset `majority_s_0.5` table now exists under `outputs/prompt_majority05_cross_dataset_rebuild_20260325/`; future replies should cite that table directly instead of falling back to `AIME`-only anecdotes.
- The metadata-only continuous-baseline pass plus the held-out top-risk bucket comparison now exist under `outputs/prompt_profile_risk_controls_20260330/`; future replies should cite that bundle rather than paraphrasing the result.
- The full-train execution object is now pinned in `docs/prompt-profile-full-train-plan-2026-04-02.md`; future handoffs should cite that file instead of restating the run contract from thread memory.
- The finished control suite did **not** find a dataset where the joint `prompt_length + effective_budget` baseline became the new winner. Future writeups should say that explicitly instead of implying the joint baseline is still unmeasured.
- Older thread notes used "rank correlation" as shorthand. Future writeups should say `Spearman rank correlation` explicitly and always name the target being ranked.
- Older notes also used "prompt-length baseline" too loosely. Future writeups should say whether this means a train-fit 1D scorer or only a raw held-out association statistic.
- Do not describe the project goal as "ranking prompts." The target-choice question is classification/regression-label selection from prompt-prefill activations; the `top 20%` bucket is only one common held-out diagnostic used to compare candidate targets.

## Known Data Gaps

- The original `LiveCodeBench` job crashed after grading and before writing its final JSON. Replay-based repair did not recover `avg_first_loop_prefix_length` exactly. That metric remains `null` in the recovered capped bundle. A fresh rerun would be required if exact prefix-length telemetry is still needed.
- The recovered `LiveCodeBench` prompt-projection artifact still lacks prompt-level correctness, so the completed bucket test cannot report prompt-level accuracy there.

## Open Implementation Issues

- This workspace still does not have a local Torch runtime / project virtualenv. Code paths are syntax-checked locally; real Torch-backed build/train execution requires the remote pilot window.

## Conditional Next Step

- The metadata-baseline pass and top-risk-bucket comparison no longer leave the binary-head choice unsettled. Keep direct `p_cap` closed unless a later contradictory slice forces it back open.

## Open Review Surfaces

- Upstream PR #7 (`Zhi0467/cot-loop`, branch `task/1773870804-prompt-profile-probe`) — prompt-profile implementation.
- Upstream PR #6 (`Zhi0467/cot-loop`, branch `task/1773451376-common-policy-refresh`) — common-policy rollout bundle.
