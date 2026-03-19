# Docs Index

Last updated: 2026-03-19 22:35 UTC

Purpose:
- Store long-lived project documentation that is not part of the main README.
- Keep each doc header dated in UTC when it is created or updated.

Current docs and references:
- Project roadmap: ../roadmap.md
- Terminal-objective design note: terminal-objective.md
- Prompt-profile implementation note: prompt-profile-probe.md
- Rollout text visualization note: rollout-text-visualization.md
- Consolidated findings: ../outputs/pr2_experiment_findings_consolidated.md
- Consolidated findings PDF: ../outputs/pr2_experiment_findings_consolidated_pdf/pr2_experiment_findings_consolidated.pdf
- Detailed reopened-round summary PDF: ../outputs/prefill_rounds_1_to_12_detailed_summary/prefill_rounds_1_to_12_detailed_summary.pdf
- Rollout-statistics module audit PDF: ../outputs/rollout_stats_module_audit/rollout_stats_module_audit.pdf
- Terminal-objective recommendation PDF (initial fixed-policy framing): ../outputs/prefill_round13_terminal_objective_report/terminal_objective_plan.pdf
- Terminal-objective follow-up PDF (cap-generalization refinement): ../outputs/prefill_round14_tail_profile_objective_report/tail_profile_objective_plan.pdf
- Terminal-objective Athena follow-up PDF: ../outputs/prefill_round15_athena_target_followup/athena_target_followup.pdf

Current live status note:
- The refreshed common-policy cross-dataset bundle is now reportable across all five datasets, with repaired `GPQA` / `MMLU-Pro` JSON-answer rows and a regenerated PDF in `../outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`. Follow the root project report (`../../projects/cot-loop-detection.md`) for the separate benchmark-style `GPQA` calibration note and the explicit reminder that recovered capped `LiveCodeBench` still leaves `avg_first_loop_prefix_length` missing after the crash.
- The current design recommendation is to move the next predictor round away from another binary loop label and away from raw `p(max_length_hit)` as the sole main target. The live objective is now a cap-robust prompt-level rollout-profile target centered on normalized-length tail exceedance plus dense normalized-length regression, while keeping max-length-hit risk as an operational eval head. The latest Athena cross-check tightened the fallback choice: if only one head ships first, use `s_0.9`, keep correctness diagnostic-only, and control the first GPQA slice so the effective budget `E` does not mostly proxy prompt length.
- The first code path for that recommendation is now in-repo: `build_probe_dataset.py` can emit prompt-level `s_t` soft targets from repeated rollouts, `train_probe.py` can train/evaluate against those soft targets, and the SLURM launcher now exposes the needed `TARGET_KIND` / `NUM_GENERATIONS` / `PROFILE_TAIL_THRESHOLD` knobs. The current implementation note is `prompt-profile-probe.md`.
- The GPQA rollout-text visualization note now has a binary-figure follow-up on top of the original pilot: the raw geometry still shows strong prompt dominance, quantified by same-prompt `5`-NN purity dropping from `0.934` to `0.444` after prompt-centering; the clearest visible binary split remains `stop` vs `length`, while exact loops still appear as a sparse subset inside the broader length-hit cloud.
- A larger-prompt GPQA rerun (`48` prompts x `4` rollouts) was attempted on 2026-03-19, but the shared GPU node queued that bounded job for `2026-03-21` and it was canceled instead of being left unattended. The current published visualization therefore remains the improved binary view on the checked-in `16 x 10` slice.
