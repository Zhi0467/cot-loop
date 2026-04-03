# Docs Index

Last updated: 2026-04-03 23:33 UTC

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
- Consolidated earlier findings PDF: ../outputs/pr2_experiment_findings_consolidated_pdf/pr2_experiment_findings_consolidated.pdf
- Rollout-statistics module audit PDF: ../outputs/rollout_stats_module_audit/rollout_stats_module_audit.pdf
- Detailed reopened-round summary PDF: ../outputs/prefill_rounds_1_to_12_detailed_summary/prefill_rounds_1_to_12_detailed_summary.pdf

Current live status:
- The current prompt-level predictor task is "predict terminal rollout statistics from prompt-prefill activations under one fixed model and decode policy," not "force everything into one binary loop label."
- The objective selector is now fixed explicitly: choose the target by held-out predictability on the target itself, not by the downstream `top 20%` loop-enrichment slice.
- Under that criterion, `mean_relative_length` is the strongest current regression target and `majority_s_0.5` is the strongest finished binary label surface, and Wangzhi has now locked that pair as the next full-train surface.
- `p_loop` still wins the old bucket diagnostic for concentrating looping prompts, but that is no longer treated as proof that it should be the main training objective.
- The note `prompt-profile-risk-screen-2026-03-30.md` is now the technical decision surface for this predictability-first correction.
- The note `prompt-profile-plain-language-2026-03-30.md` is the collaborator-facing explanation of the same correction in plain words.
- The repo-root note `../understand-where-loop-and-max-length-come-from.md` is the actual OLMo progression plan on the older rollout-statistics module: reuse the Qwen3 collector/report bundle, collect the same metric family, and test where degenerate rollouts enter along base -> SFT -> RLVR.
- The docs note `understand-where-loop-and-max-length-come-from.md` is only the background definitions appendix for saved `loop`, prompt-profile `cap_hit` / `p_cap`, rollout-stat `max_length_hit`, and `majority_s_0.5`.
- The locked pair now has a canonical execution note and PDF on disk, so the next step is the run itself rather than another planning pass.
- The run surface is now executable rather than purely documentary: `scripts/run_prompt_profile_full_train.py` is the canonical launcher, and `scripts/summarize_prompt_profile_full_train.py` is the canonical post-run ledger for the locked pair plus metadata controls.
- The reset note `thread-reset-2026-03-25.md` is now the correct restart surface for Slack follow-up. It captures the collaborator's recent corrections, the proved-vs-unproved ledger, and the exact next work order.
- `roadmap.md` is the chronological experiment log; `backlog.md` now carries the next objective run under the corrected selection rule.
- `LiveCodeBench` is no longer pending, but its recovered projection artifact still lacks prompt-level accuracy.
