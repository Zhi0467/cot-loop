# CoT Loop Detection Backlog

Last updated: 2026-04-05 00:54 UTC

## Immediate Next Experiments

- The canonical prompt-profile regression lane is now backed by both the original locked run and a fresh rerun on the current branch.
  - First citation for the current regression object:
    - `docs/prompt-profile-natural-regression-rerun-2026-04-05.md`
    - `outputs/prompt_profile_natural_regression_rerun_20260405/prompt_profile_natural_regression_rerun_20260405.pdf`
  - Earlier background notes that the rerun verifies rather than replaces:
    - `docs/prompt-profile-full-train-results-2026-04-04.md`
    - `docs/prompt-profile-full-surface-update-2026-04-04.md`
  - What the rerun proved:
    - Slurm `2215` retrained the regression lane only, on the natural prompt-disjoint train/test split with natural sampling, from the current PR branch.
    - Its copied summary matched the original locked `2043` regression ledger exactly:
      - max absolute difference `0.0`
      - no movement in prompt-only metadata baselines
      - no movement in `ensemble` or `last_layer`
      - no movement at either `best_loss` or `best_rank`
  - Why this is the canonical object now:
    - Wangzhi later rejected train balancing for `mean_relative_length`, since it is a continuous target and does not concern a binary label.
    - The natural-split / natural-sampler regression lane is therefore not just preferred in principle; it is now re-run and verified on the current branch.
  - Main natural regression read at frozen `best_loss`:
    - screening (`top_20p_capture`): ensemble beats prompt length on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`
    - calibration (`RMSE`): ensemble beats prompt length only on `GPQA` and `LiveCodeBench`
    - overall read: `mean_relative_length` stays mixed and should be reported screening-first, not as a blanket activation-lift win
  - If prompt-profile regression is pushed further, the next honest follow-up is tuning on this same natural surface:
    - small layer-subset sweep for the ensemble
    - small capacity sweep
    - keep `top_20p_capture` primary and `RMSE` secondary
- The balanced-regression notes are now provenance-only side analyses, not the regression deliverable.
  - `docs/prompt-profile-balanced-regression-2026-04-04.md`
    - first mistaken rerun that downsampled the regression train split to the balanced binary subset
  - `docs/prompt-profile-balanced-regression-corrected-2026-04-04.md`
    - second rerun that restored the full train counts but still used a balanced sampler from `majority_s_0.5`
  - These notes are still useful for explaining the count drop and for sensitivity analysis, but they should not be cited as the default regression training object going forward.
- The corrected OLMo degeneration-origin package now exists and should be cited directly.
  - Result note: `docs/olmo-degeneration-origin-audit-2026-04-04.md`
  - Result PDF: `outputs/olmo_degeneration_origin_audit_20260404/olmo_degeneration_origin_audit_20260404.pdf`
  - Main repairs:
    - `RLVR / MMLU-Pro = 0 / 80` was a grader bug on relaxed terminal forms such as `{"answer": I}`;
    - OLMo now has a model-native `LiveCodeBench` path instead of the old silent Qwen-wrapper fallback.
  - Main corrected OLMo3 read:
    - bounded OLMo3 SFT now lands at `34 / 80` on `MMLU-Pro` with `1 / 80` loop and `1 / 80` max hit;
    - bounded OLMo3 RLVR now lands at `27 / 80` on `MMLU-Pro` with `0 / 80` loops and `0 / 80` max hits;
    - bounded OLMo3 `LiveCodeBench` is now `6 / 80` for SFT with `1 / 80` loop and `22 / 80` for RLVR with `0 / 80` loops.
  - Main OLMo2 fallback read:
    - heavy loop / cap mass in base;
    - reduced but still present mass in SFT;
    - much smaller mass in `RLVR1` / instruct.
  - What is still open:
    - decide whether the next honest follow-up is a larger bounded OLMo2 slice or a smaller targeted OLMo3 base comparison;
    - do not reopen interface debugging unless a new row breaks the corrected surface again.
- The combined April prompt-profile report is now background context, not the current regression deliverable.
  - Result note: `docs/prompt-profile-full-surface-update-2026-04-04.md`
  - Result PDF: `outputs/prompt_profile_full_surface_update_20260404/prompt_profile_full_surface_update_20260404.pdf`
  - It puts the locked full-train report and the balanced binary capacity rerun back into one self-contained artifact.
  - It makes the object split explicit:
    - regression `mean_relative_length` stays on the locked full-train surface
    - the balanced rerun section in that report changes only the binary `majority_s_0.5` head
    - later balanced-regression side analyses do not replace that natural regression lane
  - Use this combined note for the natural regression plus balanced-binary split, not for the superseded balanced-regression detour.
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
