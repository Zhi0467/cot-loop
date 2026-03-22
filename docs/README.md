# Docs Index

Last updated: 2026-03-22 07:30 UTC

Purpose:
- Store long-lived project documentation that is not part of the main README.
- Keep each doc header dated in UTC when it is created or updated.

Current docs and references:
- Project roadmap: ../roadmap.md
- Terminal-objective design note: terminal-objective.md
- Prompt-profile implementation note: prompt-profile-probe.md
- Prompt-level projection note: prompt-profile-projection.md
- Prefill-activation visualization note: prefill-activation-visualization.md
- Rollout text visualization note: rollout-text-visualization.md
- Consolidated findings: ../outputs/pr2_experiment_findings_consolidated.md
- Consolidated findings PDF: ../outputs/pr2_experiment_findings_consolidated_pdf/pr2_experiment_findings_consolidated.pdf
- Detailed reopened-round summary PDF: ../outputs/prefill_rounds_1_to_12_detailed_summary/prefill_rounds_1_to_12_detailed_summary.pdf
- Rollout-statistics module audit PDF: ../outputs/rollout_stats_module_audit/rollout_stats_module_audit.pdf
- Terminal-objective recommendation PDF (initial fixed-policy framing): ../outputs/prefill_round13_terminal_objective_report/terminal_objective_plan.pdf
- Terminal-objective follow-up PDF (cap-generalization refinement): ../outputs/prefill_round14_tail_profile_objective_report/tail_profile_objective_plan.pdf
- Terminal-objective Athena follow-up PDF: ../outputs/prefill_round15_athena_target_followup/athena_target_followup.pdf
- GPQA prompt-profile recommendation PDF: ../outputs/gpqa_prompt_profile_recommendation_20260321/gpqa_prompt_profile_recommendation.pdf
- Prompt-profile objective refresh PDF: ../outputs/p_loop_objective_recommendation_20260322/p_loop_objective_recommendation.pdf

Current live status note:
- The refreshed common-policy cross-dataset bundle is now reportable across all five datasets, with repaired `GPQA` / `MMLU-Pro` JSON-answer rows and a regenerated PDF in `../outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`. Follow the root project report (`../../projects/cot-loop-detection.md`) for the separate benchmark-style `GPQA` calibration note and the explicit reminder that recovered capped `LiveCodeBench` still leaves `avg_first_loop_prefix_length` missing after the crash.
- The current design recommendation is to move the next predictor round away from another binary loop label and away from raw `p(max_length_hit)` as the sole main target. The live objective is now a prompt-level rollout-profile target trained directly on repeated-rollout labels, but the recommendation is split into two tiers: `mean_relative_length` is the best current shipped head for usefulness on the saved `GPQA` archive, while `p_loop` remains the cleaner loop-prox study head and should stay in the evaluation set rather than being dropped.
- The first code path for that recommendation is now in-repo: `build_probe_dataset.py` can emit prompt-level probability targets for `p_loop`, `p_cap`, or `s_t` from repeated rollouts, can also emit `mean_relative_length` regression labels, the builder has task-aware prompt formatting for `GPQA` / `MMLU-Pro`, `train_probe.py` can train/evaluate against both continuous target kinds, the builder writes one reusable repeated-rollout archive to `diagnostics/prompt_rollout_archive.jsonl`, and `scripts/relabel_prompt_profile_dataset.py` can relabel a finished prompt-profile dataset onto a new prompt-level target without rerolling or re-extracting activations. The current implementation note is `prompt-profile-probe.md`.
- The newest recommendation note now has one more qualification on top of the Athena-backed objective refresh. The repaired archive geometry still favors `p_loop` over thresholded tail labels as the cleaner target, but the direct same-archive GPQA retrain showed that usefulness depends on selection rule: `p_loop` shows ranking signal only at early epochs under the current pilot, while `mean_relative_length` is currently the more stable useful head.
- The first GPQA prompt-profile pilot was staged on 2026-03-19 with a local `gpqa_diamond.csv` mirror and the requested `temperature=0.2`, `n=10`, per-layer-ensemble setup, but the remote run did not start because `wth-gpu-01` was fully occupied by another user's 8-GPU Slurm job through the currently scheduled end time `2026-03-21 20:18 UTC`. The submission was canceled rather than left pending unattended.
- The resumed 2026-03-21 pilot on GPUs `0-2` is no longer only a feasibility note: it finished end to end in `37m22s`, wrote the prompt-profile archive plus probe checkpoints, and showed that runtime is acceptable on the 3-GPU path. A same-archive `mean_relative_length` retrain also landed immediately afterward, so the current status is tighter still: the remaining issue is not rollout/runtime plumbing but predictor quality versus trivial baselines on a too-small `48`-prompt slice.
- The GPQA rollout-text visualization note now has a binary-figure follow-up on top of the original pilot: the raw geometry still shows strong prompt dominance, quantified by same-prompt `5`-NN purity dropping from `0.934` to `0.444` after prompt-centering; the clearest visible binary split remains `stop` vs `length`, while exact loops still appear as a sparse subset inside the broader length-hit cloud.
- A larger-prompt GPQA rerun (`48` prompts x `4` rollouts) was attempted on 2026-03-19, but the shared GPU node queued that bounded job for `2026-03-21` and it was canceled instead of being left unattended. The current published visualization therefore remains the improved binary view on the checked-in `16 x 10` slice.
- A corrected activation-based GPQA visualization now exists on top of the finished 2026-03-21 prompt-profile pilot. It uses the saved prefill feature shards plus the rollout/profile JSONLs rather than rollout text, projects the final-layer last-prefill-token activations with one shared PCA plane, and shows that correctness is the strongest large-scale gradient while `max_length` risk appears in several prompt islands rather than one clean failure lobe.
- There is now a separate prompt-level projection path for the repeated-rollout prompt-profile bundles. It keeps one point per prompt, can recolor that same plane by prompt-majority labels and threshold-derived rates (`s_0.5`, `s_0.6`, `s_0.9`), and adds a quantitative separability table from one unsupervised cluster fit on the 2D plane. The implementation note is `prompt-profile-projection.md`.
- The old `2`-GPU all-dataset queue was canceled to match the newer `1`-GPU ceiling from the thread. The corrected serial visualization chain now runs through `slurm/run_prompt_profile_projection.sbatch`; the first GPQA validation export already shows that correctness aligns with the prompt clusters much more than prompt-majority loop / cap labels, and that `s_0.5` and `s_0.6` are nearly indistinguishable on that saved slice.
