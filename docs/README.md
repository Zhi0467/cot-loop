# Docs Index

Last updated: 2026-03-30 03:10 UTC

Purpose:
- Store long-lived project documentation that is not part of the main README.
- Keep each doc header dated in UTC when it is created or updated.

Core docs:
- Project roadmap: ../roadmap.md
- Open experiment ledger / next runs: ../backlog.md
- Prompt-profile implementation path: prompt-profile-probe.md
- Prompt-profile evaluation contract: prompt-profile-eval-contract.md
- Prompt-profile risk-screen decision: prompt-profile-risk-screen-2026-03-30.md
- Thread reset / new-thread handoff: thread-reset-2026-03-25.md
- Prompt-profile projection/export path: prompt-profile-projection.md
- Prefill-activation visualization note: prefill-activation-visualization.md

Key outputs:
- Common-policy rollout report PDF: ../outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/qwen3_1p7b_cross_dataset_rollout_report.pdf
- Prompt-majority control note PDF: ../outputs/prompt_majority_05_all_datasets_recommendation_20260321/prompt_majority_05_all_datasets_recommendation.pdf
- `p_loop` objective note PDF: ../outputs/p_loop_objective_recommendation_20260322/p_loop_objective_recommendation.pdf
- Two-head recommendation PDF: ../outputs/two_head_prompt_profile_recommendation_20260322/two_head_prompt_profile_recommendation.pdf
- Build-recipe PDF: ../outputs/prompt_profile_build_recipe_20260323/prompt_profile_build_recipe.pdf
- Prompt-profile risk-control bundle: ../outputs/prompt_profile_risk_controls_20260330/
- Consolidated earlier findings PDF: ../outputs/pr2_experiment_findings_consolidated_pdf/pr2_experiment_findings_consolidated.pdf
- Rollout-statistics module audit PDF: ../outputs/rollout_stats_module_audit/rollout_stats_module_audit.pdf
- Detailed reopened-round summary PDF: ../outputs/prefill_rounds_1_to_12_detailed_summary/prefill_rounds_1_to_12_detailed_summary.pdf

Current live status:
- The current prompt-level predictor task is "predict terminal rollout statistics from prompt-prefill activations under one fixed model and decode policy," not "force everything into one binary loop label."
- The main prompt-level screening target is now `p_loop`; the first real continuous metadata baselines plus the held-out top-risk-bucket test are complete, and `p_loop` is the only score that consistently concentrates loop-heavy, cap-heavy, low-accuracy prompts across the saved datasets.
- `mean_relative_length` remains useful as a secondary utility / budget-consumption head, not as the main degeneracy screen.
- `majority_s_0.5` is still worth keeping as a control and possible cheap auxiliary screen, but the finished bucket test now makes it a geometry-heavy control rather than the main target.
- The new note `prompt-profile-risk-screen-2026-03-30.md` is the current decision surface for the objective choice and its caveats.
- The reset note `thread-reset-2026-03-25.md` is now the correct restart surface for Slack follow-up. It captures the collaborator's recent corrections, the proved-vs-unproved ledger, and the exact next work order.
- `roadmap.md` is the chronological experiment log; `backlog.md` now holds the next prospective follow-ups after the objective decision rather than the already-finished metadata/bucket gap.
- `LiveCodeBench` is no longer pending. The recovered follow-up plus the finished bucket test reinforced the final ordering: `p_loop` as the main screen, `mean_relative_length` as the secondary utility head.
