# Docs Index

Last updated: 2026-03-16 14:22 UTC

Purpose:
- Store long-lived project documentation that is not part of the main README.
- Keep each doc header dated in UTC when it is created or updated.

Current docs and references:
- Project roadmap: ../roadmap.md
- Consolidated findings: ../outputs/pr2_experiment_findings_consolidated.md
- Consolidated findings PDF: ../outputs/pr2_experiment_findings_consolidated_pdf/pr2_experiment_findings_consolidated.pdf
- Detailed reopened-round summary PDF: ../outputs/prefill_rounds_1_to_12_detailed_summary/prefill_rounds_1_to_12_detailed_summary.pdf
- Rollout-statistics module audit PDF: ../outputs/rollout_stats_module_audit/rollout_stats_module_audit.pdf

Current live status note:
- The refreshed common-policy cross-dataset bundle is now reportable across all five datasets, with repaired `GPQA` / `MMLU-Pro` JSON-answer rows and a regenerated PDF in `../outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`. Follow the root project report (`../../projects/cot-loop-detection.md`) for the separate benchmark-style `GPQA` calibration note and the explicit reminder that recovered capped `LiveCodeBench` still leaves `avg_first_loop_prefix_length` missing after the crash.
