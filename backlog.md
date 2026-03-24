# CoT Loop Detection Backlog

## Next Experiment: Joint Metadata Baseline

- The five-dataset same-archive table is now complete; `LiveCodeBench` did not change the ranking (`mean_relative_length` first, `p_loop` second, `loop_budget_share` auxiliary-only).
- The main open measurement gap is the missing joint `prompt_length + effective_budget` baseline. The current saved summaries only include the separate one-variable baselines.
- Once that joint baseline exists, future head changes should be judged against it rather than against prompt length alone.

## Known Data Gaps

- The original `LiveCodeBench` job crashed after grading and before writing its final JSON. Replay-based repair did not recover `avg_first_loop_prefix_length` exactly. That metric remains `null` in the recovered capped bundle. A fresh rerun would be required if exact prefix-length telemetry is still needed.

## Open Implementation Issues

- This workspace still does not have a local Torch runtime / project virtualenv. Code paths are syntax-checked locally; real Torch-backed build/train execution requires the remote pilot window.

## Conditional Next Steps

- If a future dataset/model variant contradicts the current head ordering enough to justify one more binary-head test, reopen direct `p_cap` next rather than another threshold sweep or `loop_budget_share`.

## Open Review Surfaces

- Upstream PR #7 (`Zhi0467/cot-loop`, branch `task/1773870804-prompt-profile-probe`) — prompt-profile implementation.
- Upstream PR #6 (`Zhi0467/cot-loop`, branch `task/1773451376-common-policy-refresh`) — common-policy rollout bundle.
