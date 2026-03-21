# Docs Index

Last updated: 2026-03-21 23:53 UTC

Purpose:
- Store long-lived project documentation that is not part of the main README.
- Keep each doc header dated in UTC when it is created or updated.

Current docs and references:
- Project roadmap: ../roadmap.md
- Terminal-objective design note: terminal-objective.md
- Prompt-profile implementation note: prompt-profile-probe.md
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

Current live status note:
- The refreshed common-policy cross-dataset bundle is now reportable across all five datasets, with repaired `GPQA` / `MMLU-Pro` JSON-answer rows and a regenerated PDF in `../outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`. Follow the root project report (`../../projects/cot-loop-detection.md`) for the separate benchmark-style `GPQA` calibration note and the explicit reminder that recovered capped `LiveCodeBench` still leaves `avg_first_loop_prefix_length` missing after the crash.
- The current design recommendation is to move the next predictor round away from another binary loop label and away from raw `p(max_length_hit)` as the sole main target. The live objective is now a prompt-level rollout-profile target trained directly on repeated-rollout labels: first `s_0.9`, then a second single-head regression on `mean_relative_length`, while keeping max-length-hit risk diagnostic-only. The leakage check remains prompt-token-count and/or effective budget `E`, not a second probe loss.
- The first code path for that recommendation is now in-repo: `build_probe_dataset.py` can emit prompt-level `s_t` soft targets from repeated rollouts, can also emit `mean_relative_length` regression labels, the builder now has task-aware prompt formatting for `GPQA` / `MMLU-Pro`, `train_probe.py` can train/evaluate against both continuous target kinds, and the builder writes one reusable repeated-rollout archive to `diagnostics/prompt_rollout_archive.jsonl`. The current implementation note is `prompt-profile-probe.md`.
- The newest recommendation note now includes the first real `GPQA` pilot read, not just the pre-pilot target argument. The update is important: the 2026-03-21 `32 / 16` pilot completed cleanly, but on that actual slice `s_0.9` collapsed exactly to `p(max_length_hit)` prompt by prompt and the trained probe was essentially constant-baseline quality. The current recommendation therefore shifts one notch: keep the prompt-profile framing, but move the next main head to `mean_relative_length` (or lower the tail threshold) rather than treating `s_0.9` as settled.
- The first GPQA prompt-profile pilot was staged on 2026-03-19 with a local `gpqa_diamond.csv` mirror and the requested `temperature=0.2`, `n=10`, per-layer-ensemble setup, but the remote run did not start because `wth-gpu-01` was fully occupied by another user's 8-GPU Slurm job through the currently scheduled end time `2026-03-21 20:18 UTC`. The submission was canceled rather than left pending unattended.
- The resumed 2026-03-21 pilot on GPUs `0-2` is no longer only a feasibility note: it finished end to end in `37m22s`, wrote the prompt-profile archive plus probe checkpoints, and showed that runtime is acceptable on the 3-GPU path. A same-archive `mean_relative_length` retrain also landed immediately afterward, so the current status is tighter still: the remaining issue is not rollout/runtime plumbing but predictor quality versus trivial baselines on a too-small `48`-prompt slice.
- The GPQA rollout-text visualization note now has a binary-figure follow-up on top of the original pilot: the raw geometry still shows strong prompt dominance, quantified by same-prompt `5`-NN purity dropping from `0.934` to `0.444` after prompt-centering; the clearest visible binary split remains `stop` vs `length`, while exact loops still appear as a sparse subset inside the broader length-hit cloud.
- A larger-prompt GPQA rerun (`48` prompts x `4` rollouts) was attempted on 2026-03-19, but the shared GPU node queued that bounded job for `2026-03-21` and it was canceled instead of being left unattended. The current published visualization therefore remains the improved binary view on the checked-in `16 x 10` slice.
- A corrected activation-based GPQA visualization now exists on top of the finished 2026-03-21 prompt-profile pilot. It uses the saved prefill feature shards plus the rollout/profile JSONLs rather than rollout text, projects the final-layer last-prefill-token activations with one shared PCA plane, and shows that correctness is the strongest large-scale gradient while `max_length` risk appears in several prompt islands rather than one clean failure lobe.
