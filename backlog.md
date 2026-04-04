# CoT Loop Detection Backlog

Last updated: 2026-04-04 21:43 UTC

## Immediate Next Experiments

- The balanced-only prompt-profile regression rerun now exists and should be cited directly.
  - Result note: `docs/prompt-profile-balanced-regression-2026-04-04.md`
  - Result PDF: `outputs/prompt_profile_balanced_regression_20260404/prompt_profile_balanced_regression_20260404.pdf`
  - Copied summary ledger: `outputs/prompt_profile_balanced_regression_20260404/remote_summary/`
  - Exact split contract:
    - reuse the saved balanced `majority_s_0.5` train/test prompt IDs from `/data/scratch/murphy/outputs/cot-loop-detection/full_train_locked_pair_20260404/`
    - relabel those same prompts with `mean_relative_length`
    - keep train balanced on the binary split and test natural
    - keep the default regression probe family (`hidden_dim=128`, `depth=1`, `dropout=0.1`)
  - Main screening read at frozen `best_loss`:
    - cross-dataset mean `top_20p_capture`: ensemble `0.344`, prompt length `0.262`, last-layer `0.257`
    - ensemble beats prompt length on `GPQA`, `MATH-500`, `MMLU-Pro`, and `LiveCodeBench`
    - `AIME` is the only non-win against the prompt-length baseline on that metric
  - Calibration caveat:
    - cross-dataset mean `RMSE`: last-layer `0.191`, ensemble `0.205`, prompt length `0.265`
    - this keeps `mean_relative_length` as a screening-first surface rather than a clean calibrated regressor
  - Two code-path fixes landed during the rerun and should be treated as part of the durable execution surface:
    - Slurm wrappers now honor explicit `CONDA_ENV` before any stale repo-local `.venv`
    - `scripts/summarize_prompt_profile_full_train.py` now summarizes regression-only reruns instead of marking them `missing_shared_archive`
  - If prompt-profile regression is pushed further, the next honest follow-up is not "rerun balanced regression again." It is a small layer-subset or capacity sweep on this same balanced regression split, while keeping `top_20p_capture` primary and `RMSE` secondary.
- Audit the weird bounded OLMo instruct bundle before scaling it or swapping model families.
  - The current bundle is under `/data/scratch/murphy/outputs/cot-loop-detection/olmo3_degeneration_origin_progression/bound8_temp0p1_gen10_ctx40960_topkneg1/`.
  - `RLVR / MMLU-Pro = 0 / 80` with mean length `6.15` is too suspicious to treat as a clean capability read.
  - `SFT / LiveCodeBench = 0 / 80` also needs the native-codegen surface checked before being treated as an ordinary rollout-success result.
  - Run the cheap audit first:
    - inspect terminal answer forms on saved `MMLU-Pro` responses with the relaxed structured parser, not only the strict JSON grader;
    - inspect `LiveCodeBench` extracted code plus native `pass@k` on the same saved bundle, not only rollout-success;
    - only after that decide whether the current OLMo 3 surface is scientifically usable.
  - If a smaller progression is still needed after the audit, the smallest public fallback is the April 2025 OLMo 2 `1B` chain (`OLMo-2-0425-1B -> OLMo-2-0425-1B-SFT -> OLMo-2-0425-1B-RLVR1 -> OLMo-2-0425-1B-Instruct`).
  - Do not describe that as a smaller OLMo 3 run: the public OLMo 3 instruct ladder is only `7B` and `32B`.
- The combined April prompt-profile report is now background context, not the current balanced-regression deliverable.
  - Result note: `docs/prompt-profile-full-surface-update-2026-04-04.md`
  - Result PDF: `outputs/prompt_profile_full_surface_update_20260404/prompt_profile_full_surface_update_20260404.pdf`
  - It puts the locked full-train report and the balanced binary capacity rerun back into one self-contained artifact.
  - It makes the object split explicit:
    - regression `mean_relative_length` stays on the locked full-train surface
    - the balanced rerun section in that report changes only the binary `majority_s_0.5` head
    - the collaborator-requested balanced regression rerun is now answered by `docs/prompt-profile-balanced-regression-2026-04-04.md`
  - For the collaborator-correct balanced-only regression question, cite the balanced regression note above instead of this combined note.
  - Main combined read:
    - regression is still mixed and should be reported screening-first with `top_20p_capture`
    - binary remains the cleaner deployment-facing head
    - current global ensemble recommendation on the balanced binary object is `h256 d1`
- The binary capacity-control sweep now exists as the latest result surface.
  - Result note: `docs/prompt-profile-binary-capacity-controls-2026-04-04.md`
  - Result PDF: `outputs/prompt_profile_binary_capacity_controls_20260404/prompt_profile_binary_capacity_controls_20260404.pdf`
  - This is the same saved April balanced binary data with three probe families: `h128 d1`, `h256 d1`, `h256 d2`.
  - Main read: if one single global ensemble surface is needed today, it should be `h256 d1`, not `h256 d2`.
  - The width-vs-depth split is now explicit:
    - for `ensemble`, width helps and added depth hurts;
    - for `last_layer`, added depth helps modestly on top of width.
  - The global ensemble recommendation is stable across checkpoint rules:
    - frozen `best_loss`: mean test `PR-AUC 0.518` for `ensemble h256 d1` versus `0.492` for `ensemble h256 d2`;
    - secondary `best_rank`: mean test `PR-AUC 0.539` for `ensemble h256 d1` versus `0.522` for `ensemble h256 d2`.
  - Threshold metrics on the natural test split remain recall-heavy on rare-positive datasets, so keep `PR-AUC` primary and treat threshold metrics as diagnostic only.
  - Proven limitation: this still does not answer which layers to keep. The next honest tuning step is a small layer-subset / view sweep on the same balanced binary data.
- The older `h256 d2` balanced binary rerun note is now intermediate evidence rather than the recommendation surface.
  - Result note: `docs/prompt-profile-binary-retrain-h256d2-2026-04-04.md`
  - Result bundle: `outputs/prompt_profile_binary_retrain_h256d2_20260404/`
  - This is still useful because it preserves the exact `2106` / `2107` depth-rerun metrics and logs.
  - But it should now be read only as the depth-only control that motivated the width-only `2108` follow-up, not as the current best-surface note.
  - The superseding recommendation surface is the capacity-controls note above.
- Interpret the finished locked full-train pass before reopening targets.
  - Result note: `docs/prompt-profile-full-train-results-2026-04-04.md`
  - Result PDF: `outputs/prompt_profile_full_train_locked_pair_20260404/prompt_profile_full_train_locked_pair_20260404.pdf`
  - Copied summary ledger: `outputs/prompt_profile_full_train_locked_pair_20260404/remote_summary/`
  - The run contract stayed fixed to the saved prompt-profile surface: `Qwen/Qwen3-1.7B`, `temperature=0.2`, `num_generations=4`, `loop_n=30`, `loop_k=20`, prompt-prefill only.
  - Aggregation rule matters: regression `ensemble` uses `mean_prob`, while binary `ensemble` uses `vote_fraction`.
  - Regression read: on held-out prompt-level screening `top_20p_capture` at the frozen `best_loss` checkpoint, ensemble beats last-layer on `4 / 5` datasets and beats the train-fit prompt-length baseline on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`, but still loses on `AIME` and `MATH-500`.
  - Binary read: ensemble `PR-AUC` beats the prompt-length baseline on all five datasets, with the clearest finished wins on `AIME`, `LiveCodeBench`, and `MMLU-Pro`.
  - Keep `best_loss` as the frozen checkpoint for this run; do not reseat the regression checkpoint after the fact just because the reporting lens moved to screening.
  - For this run, keep `top_20p_capture` as the operational regression metric, `RMSE` as calibration context, and `Spearman` only as a tertiary monotone-ordering diagnostic over aligned held-out prompt pairs.
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

This should be similar to our previous experiments on training probes on loop label, only with different labels this time; reuse as much as possible. Come back with similar reports as we did with the previous loop label experiments.

## Fixed Experimental Surface

- Keep the predictor input to prompt-prefill activations only.
- Stay in-distribution and prompt-disjoint.
- Keep one selected layer or per-layer independent probes with late aggregation.
- Keep the decode policy fixed at `temperature = 0.2` while settling the objective question.

## Measurement And Reporting Gaps

- The explicit cross-dataset `majority_s_0.5` table now exists under `outputs/prompt_majority05_cross_dataset_rebuild_20260325/`; future replies should cite that table directly instead of falling back to `AIME`-only anecdotes.
- The metadata-only continuous-baseline pass plus the held-out top-risk bucket comparison now exist under `outputs/prompt_profile_risk_controls_20260330/`; future replies should cite that bundle rather than paraphrasing the result.
- The full-train execution object is now pinned in `docs/prompt-profile-full-train-plan-2026-04-02.md`; future handoffs should cite that file instead of restating the run contract from thread memory.
- The locked pair now has executable command surfaces too: `scripts/run_prompt_profile_full_train.py` for the run itself and `scripts/summarize_prompt_profile_full_train.py` for the cross-dataset ledger. Future handoffs should point to those scripts instead of recopying the manual command sequence.
- The first locked full-train result bundle now exists under `outputs/prompt_profile_full_train_locked_pair_20260404/`; future replies should cite that ledger directly instead of falling back to the early GPQA/AIME-only Slack checkpoints.
- The finished control suite did **not** find a dataset where the joint `prompt_length + effective_budget` baseline became the new winner. Future writeups should say that explicitly instead of implying the joint baseline is still unmeasured.
- On the fixed `max_tokens=30000` full-train surface, the `effective_budget` control is constant. Future writeups should say that explicitly instead of treating it as an independent moving metadata signal in this run.
- Older thread notes used "rank correlation" as shorthand. Future writeups should say `Spearman rank correlation` explicitly and always name the target being ranked.
- Older notes also used "prompt-length baseline" too loosely. Future writeups should say whether this means a train-fit 1D scorer or only a raw held-out association statistic.
- For the locked full-train run specifically, "metadata baseline" means prompt-only scorers on `prompt_token_count` and `effective_max_tokens`; since `effective_max_tokens=30000` is constant here, the only nontrivial baseline feature is prompt length.
- Future writeups must say whether `mean_relative_length` is being used as a calibrated regression target or as a screening score. The same frozen run supports both reads, but they lead to different headline claims.
- Do not describe the project goal as "ranking prompts." The target-choice question is classification/regression-label selection from prompt-prefill activations; the `top 20%` bucket is only one common held-out diagnostic used to compare candidate targets.

## Known Data Gaps

- The original `LiveCodeBench` job crashed after grading and before writing its final JSON. Replay-based repair did not recover `avg_first_loop_prefix_length` exactly. That metric remains `null` in the recovered capped bundle. A fresh rerun would be required if exact prefix-length telemetry is still needed.
- The recovered `LiveCodeBench` prompt-projection artifact still lacks prompt-level correctness, so the completed bucket test cannot report prompt-level accuracy there.

## Conditional Next Step

- The metadata-baseline pass and top-risk-bucket comparison no longer leave the binary-head choice unsettled. Keep direct `p_cap` closed unless a later contradictory slice forces it back open.

Use GPU node for reruns of this surface, 2 GPUs.
