# CoT Loop Detection Backlog

Last updated: 2026-03-30 03:10 UTC

## Immediate Next Experiments

- Freeze the new objective choice prospectively.
  - Run one fresh prompt-disjoint `p_loop` fit on a representative hard slice with the architecture/view/checkpoint rule fixed up front, so the new objective recommendation is not only retrospective on saved bundles.
  - Keep the output note explicit that this is the first prospective confirmation after the retrospective bucket comparison.
- Keep `mean_relative_length` as a secondary head only where it still buys utility.
  - If a deployment path wants both a degeneracy screen and a budget / difficulty proxy, train `mean_relative_length` beside `p_loop`.
  - Do not let that secondary use silently turn back into the main objective.
- Recover `LiveCodeBench` prompt-level accuracy only if the 5/5 accuracy table becomes operationally necessary.
  - The current recovered projection surface still lacks prompt-level correctness, so the completed bucket test is `4 / 5` datasets for accuracy even though the loop / cap comparisons are complete.
- Reopen direct `p_cap` only on contradiction.
  - The finished metadata + bucket pass no longer leaves the binary choice unsettled.
  - Only reopen `p_cap` if a later slice shows cap-hit isolation matters beyond what `p_loop` already surfaces.

## Fixed Experimental Surface

- Keep the predictor input to prompt-prefill activations only.
- Stay in-distribution and prompt-disjoint.
- Keep one selected layer or per-layer independent probes with late aggregation.
- Keep the decode policy fixed at `temperature = 0.2` while settling the objective question.

## Measurement And Reporting Gaps

- The explicit cross-dataset `majority_s_0.5` table now exists under `outputs/prompt_majority05_cross_dataset_rebuild_20260325/`; future replies should cite that table directly instead of falling back to `AIME`-only anecdotes.
- The metadata-only continuous-baseline pass plus the held-out top-risk bucket comparison now exist under `outputs/prompt_profile_risk_controls_20260330/`; future replies should cite that bundle rather than paraphrasing the result.
- The finished control suite did **not** find a dataset where the joint `prompt_length + effective_budget` baseline became the new winner. Future writeups should say that explicitly instead of implying the joint baseline is still unmeasured.
- Older thread notes used "rank correlation" as shorthand. Future writeups should say `Spearman rank correlation` explicitly and always name the target being ranked.
- Older notes also used "prompt-length baseline" too loosely. Future writeups should say whether this means a train-fit 1D scorer or only a raw held-out association statistic.

## Known Data Gaps

- The original `LiveCodeBench` job crashed after grading and before writing its final JSON. Replay-based repair did not recover `avg_first_loop_prefix_length` exactly. That metric remains `null` in the recovered capped bundle. A fresh rerun would be required if exact prefix-length telemetry is still needed.
- The recovered `LiveCodeBench` prompt-projection artifact still lacks prompt-level correctness, so the completed bucket test cannot report prompt-level accuracy there.

## Open Implementation Issues

- This workspace still does not have a local Torch runtime / project virtualenv. Code paths are syntax-checked locally; real Torch-backed build/train execution requires the remote pilot window.

## Conditional Next Step

- The metadata-baseline pass and top-risk-bucket comparison no longer leave the binary-head choice unsettled. Keep direct `p_cap` closed unless a later contradictory slice forces it back open.

## Open Review Surfaces

- Upstream PR #7 (`Zhi0467/cot-loop`, branch `task/1773870804-prompt-profile-probe`) — prompt-profile implementation.
- Upstream PR #6 (`Zhi0467/cot-loop`, branch `task/1773451376-common-policy-refresh`) — common-policy rollout bundle.
