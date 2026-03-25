# Docs Index

Last updated: 2026-03-25 00:23 UTC

Purpose:
- Store long-lived project documentation that is not part of the main README.
- Keep each doc header dated in UTC when it is created or updated.

Core docs:
- Project roadmap: ../roadmap.md
- Prompt-profile implementation path: prompt-profile-probe.md
- Prompt-profile evaluation contract: prompt-profile-eval-contract.md
- Prompt-profile projection/export path: prompt-profile-projection.md
- Prefill-activation visualization note: prefill-activation-visualization.md

Key outputs:
- Common-policy rollout report PDF: ../outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/qwen3_1p7b_cross_dataset_rollout_report.pdf
- Prompt-majority control note PDF: ../outputs/prompt_majority_05_all_datasets_recommendation_20260321/prompt_majority_05_all_datasets_recommendation.pdf
- `p_loop` objective note PDF: ../outputs/p_loop_objective_recommendation_20260322/p_loop_objective_recommendation.pdf
- Two-head recommendation PDF: ../outputs/two_head_prompt_profile_recommendation_20260322/two_head_prompt_profile_recommendation.pdf
- Build-recipe PDF: ../outputs/prompt_profile_build_recipe_20260323/prompt_profile_build_recipe.pdf
- Consolidated earlier findings PDF: ../outputs/pr2_experiment_findings_consolidated_pdf/pr2_experiment_findings_consolidated.pdf
- Rollout-statistics module audit PDF: ../outputs/rollout_stats_module_audit/rollout_stats_module_audit.pdf
- Detailed reopened-round summary PDF: ../outputs/prefill_rounds_1_to_12_detailed_summary/prefill_rounds_1_to_12_detailed_summary.pdf

Current live status:
- The current prompt-level predictor task is "predict terminal rollout statistics from prompt-prefill activations under one fixed model and decode policy," not "force everything into one binary loop label."
- The default bundle remains `mean_relative_length` as the main useful score plus `p_loop` as the cleaner failure-prox companion.
- `majority_s_0.5` is still worth keeping as a control and possible cheap degenerate-prompt screen, but it is too prompt-length-shaped on `AIME` to be the main activation-lift claim.
- The evaluation-contract note now makes one ambiguity explicit: the binary majority table already has a true one-feature prompt-length scorer, while the current five-dataset continuous-head table still only records raw prompt-length association. A trained metadata-only continuous baseline suite is still the next missing measurement piece.
- `LiveCodeBench` is no longer pending. The recovered follow-up reinforced rather than changed the `mean_relative_length` plus `p_loop` ranking.
