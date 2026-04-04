# Docs Index

Last updated: 2026-04-04 06:31 UTC

Purpose:
- Store long-lived project documentation that is not part of the main README.
- Keep each doc header dated in UTC when it is created or updated.

Core docs:
- Project roadmap: ../roadmap.md
- Open experiment ledger / next runs: ../backlog.md
- Degeneracy-origin rollout-stat plan: ../understand-where-loop-and-max-length-come-from.md
- Prompt-profile implementation path: prompt-profile-probe.md
- Prompt-profile evaluation contract: prompt-profile-eval-contract.md
- Prompt-profile risk-screen decision: prompt-profile-risk-screen-2026-03-30.md
- Prompt-profile plain-language note: prompt-profile-plain-language-2026-03-30.md
- Prompt-profile full-train plan: prompt-profile-full-train-plan-2026-04-02.md
- Prompt-profile full-train result note: prompt-profile-full-train-results-2026-04-04.md
- Prompt-profile binary retrain note: prompt-profile-binary-retrain-h256d2-2026-04-04.md
- Prompt-profile binary capacity-controls note: prompt-profile-binary-capacity-controls-2026-04-04.md
- Thread reset / new-thread handoff: thread-reset-2026-03-25.md
- Prompt-profile projection/export path: prompt-profile-projection.md
- Prefill-activation visualization note: prefill-activation-visualization.md
- Loop / max-length definitions appendix: understand-where-loop-and-max-length-come-from.md

Key outputs:
- Common-policy rollout report PDF: ../outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/qwen3_1p7b_cross_dataset_rollout_report.pdf
- Prompt-majority control note PDF: ../outputs/prompt_majority_05_all_datasets_recommendation_20260321/prompt_majority_05_all_datasets_recommendation.pdf
- `p_loop` objective note PDF: ../outputs/p_loop_objective_recommendation_20260322/p_loop_objective_recommendation.pdf
- Two-head recommendation PDF: ../outputs/two_head_prompt_profile_recommendation_20260322/two_head_prompt_profile_recommendation.pdf
- Build-recipe PDF: ../outputs/prompt_profile_build_recipe_20260323/prompt_profile_build_recipe.pdf
- Prompt-profile risk-control bundle: ../outputs/prompt_profile_risk_controls_20260330/
- Plain-language objective PDF: ../outputs/prompt_profile_plain_language_20260330/prompt_profile_plain_language_20260330.pdf
- Full-train plan PDF: ../outputs/prompt_profile_full_train_plan_20260402/prompt_profile_full_train_plan_20260402.pdf
- Full-train result bundle: ../outputs/prompt_profile_full_train_locked_pair_20260404/
- Full-train result PDF: ../outputs/prompt_profile_full_train_locked_pair_20260404/prompt_profile_full_train_locked_pair_20260404.pdf
- Binary retrain result bundle: ../outputs/prompt_profile_binary_retrain_h256d2_20260404/
- Binary retrain result PDF: ../outputs/prompt_profile_binary_retrain_h256d2_20260404/prompt_profile_binary_retrain_h256d2_20260404.pdf
- Binary capacity-controls bundle: ../outputs/prompt_profile_binary_capacity_controls_20260404/
- Binary capacity-controls PDF: ../outputs/prompt_profile_binary_capacity_controls_20260404/prompt_profile_binary_capacity_controls_20260404.pdf
- Consolidated earlier findings PDF: ../outputs/pr2_experiment_findings_consolidated_pdf/pr2_experiment_findings_consolidated.pdf
- Rollout-statistics module audit PDF: ../outputs/rollout_stats_module_audit/rollout_stats_module_audit.pdf
- Detailed reopened-round summary PDF: ../outputs/prefill_rounds_1_to_12_detailed_summary/prefill_rounds_1_to_12_detailed_summary.pdf

Current live status:
- The current prompt-level predictor task is "predict terminal rollout statistics from prompt-prefill activations under one fixed model and decode policy," not "force everything into one binary loop label."
- The objective selector is now fixed explicitly: choose the target by held-out predictability on the target itself, not by the downstream `top 20%` loop-enrichment slice.
- Under that criterion, `mean_relative_length` is the strongest current regression target and `majority_s_0.5` is the strongest finished binary label surface, and Wangzhi locked that pair for the first full-train pass.
- `p_loop` still wins the old bucket diagnostic for concentrating looping prompts, but that is no longer treated as proof that it should be the main training objective.
- The note `prompt-profile-risk-screen-2026-03-30.md` is now the technical decision surface for this predictability-first correction.
- The note `prompt-profile-plain-language-2026-03-30.md` is the collaborator-facing explanation of the same correction in plain words.
- The repo-root note `../understand-where-loop-and-max-length-come-from.md` is the actual OLMo progression plan on the older rollout-statistics module: reuse the Qwen3 collector/report bundle, collect the same metric family, and test where degenerate rollouts enter along base -> SFT -> RLVR.
- The docs note `understand-where-loop-and-max-length-come-from.md` is only the background definitions appendix for saved `loop`, prompt-profile `cap_hit` / `p_cap`, rollout-stat `max_length_hit`, and `majority_s_0.5`.
- The locked pair now has both the execution note and the finished first-run result note on disk.
- The older `h256 d2` binary retrain note is still on disk, but it is now intermediate only:
  - it preserves the exact `2106` / `2107` depth-rerun record and raw remote metrics
  - it should not be paraphrased as the current best-surface recommendation anymore
- The follow-up capacity-control note is now the current binary tuning surface:
  - same balanced binary data, now compared across `h128 d1`, `h256 d1`, and `h256 d2`
  - the clean current read is that width helps the ensemble, extra depth hurts the ensemble, and extra depth only helps the `last_layer` view modestly
  - under the frozen `best_loss` rule, `ensemble h256 d1` is the best single global surface (`0.518` mean test `PR-AUC`)
  - under the secondary `best_rank` rule, the ensemble ordering stays the same (`0.539 > 0.522 > 0.474`), so the global ensemble choice is stable
  - threshold metrics on the natural test split remain recall-heavy on rare-positive datasets, so `PR-AUC` stays primary and threshold metrics stay diagnostic
  - if one single global binary surface is needed today, it should be `ensemble h256 d1`
- The run surface is now executable rather than purely documentary: `scripts/run_prompt_profile_full_train.py` is the canonical launcher, and `scripts/summarize_prompt_profile_full_train.py` is the canonical post-run ledger for the locked pair plus metadata controls.
- The first locked full-train pass is now complete. The repo-facing result is split cleanly:
  - regression `mean_relative_length`: ensemble beats `last_layer`, but the train-fit prompt-length baseline still wins on `AIME`, `MATH-500`, and `MMLU-Pro`;
  - binary `majority_s_0.5`: ensemble `PR-AUC` beats the prompt-length baseline on all five datasets, with the clearest finished wins on `AIME`, `LiveCodeBench`, and `MMLU-Pro`.
- The reset note `thread-reset-2026-03-25.md` is now the correct restart surface for Slack follow-up. It captures the collaborator's recent corrections, the proved-vs-unproved ledger, and the exact next work order.
- `roadmap.md` is the chronological experiment log; `backlog.md` now carries the post-run interpretation object under the corrected selection rule.
- `LiveCodeBench` is no longer pending, but its recovered projection artifact still lacks prompt-level accuracy.
