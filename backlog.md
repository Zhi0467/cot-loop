# CoT Loop Detection Backlog

## Active Experiment: LiveCodeBench Completion

- `LiveCodeBench` is the only unfinished dataset in the cross-dataset validation suite.
- The current rollout job is `1668` on 2 GPUs with `MAX_NUM_SEQS=16`, running the `640 / 160`, `n = 4` surface.
- The dependent follow-up job `1693` (behind `afterok:1668`) will train majority controls, relabel direct heads (`p_loop`, `mean_relative_length`, `loop_budget_share`), run fits, and write `lcb_followup_summary.json`.
- As of 2026-03-23 06:42 UTC, `1668` had reached 332 / 800 prompts total (176/400 on dp-rank 0, 156/400 on dp-rank 1) after 2h52m.

## Known Data Gaps

- The original `LiveCodeBench` job crashed after grading and before writing its final JSON. Replay-based repair did not recover `avg_first_loop_prefix_length` exactly. That metric remains `null` in the recovered capped bundle. A fresh rerun would be required if exact prefix-length telemetry is still needed.

## Open Implementation Issues

- This workspace still does not have a local Torch runtime / project virtualenv. Code paths are syntax-checked locally; real Torch-backed build/train execution requires the remote pilot window.

## Conditional Next Steps

- If `LiveCodeBench` contradicts the current head ordering enough to justify one more binary-head test, reopen direct `p_cap` next rather than another threshold sweep or `loop_budget_share`.

## Open Review Surfaces

- Upstream PR #7 (`Zhi0467/cot-loop`, branch `task/1773870804-prompt-profile-probe`) — prompt-profile implementation.
- Upstream PR #6 (`Zhi0467/cot-loop`, branch `task/1773451376-common-policy-refresh`) — common-policy rollout bundle.
