# CoT Loop Detection Backlog

Last updated: 2026-03-25 10:02 UTC

## Immediate Next Experiments

- Close the unresolved cross-dataset degeneracy question before making another objective claim.
  - On the existing held-out archives, rank prompts by `majority_s_0.5`, `p_loop`, `mean_relative_length`, and the metadata baselines below.
  - For the top predicted-risk `20%` prompts under each score, measure actual loop rate, actual max-length-hit rate, and actual accuracy.
  - This is the first experiment that directly answers whether the threshold label is actually catching degenerate prompts across datasets or only catching prompt geometry.
  - Until this exists, do not claim that `majority_s_0.5` is the right cross-dataset operational screen.
- Add the first true metadata-only predictor pass for the continuous heads.
  - Fit `prompt_length` only, `effective_budget` only, and `prompt_length + effective_budget` on the same train split used by the activation probes.
  - Evaluate them on the same held-out metrics as the probe heads: `Spearman`, `top-20% capture`, and `Brier` or `MSE` where relevant.
  - Keep the docs explicit that this is the first real metadata-only predictor pass for the continuous heads; the current saved table is still only raw association, not a trained baseline.
- Check whether the soft labels actually isolate degenerate prompts.
  - For `majority_s_0.5`, `p_loop`, and `mean_relative_length`, compare the predicted top-risk 20% prompts on empirical loop rate, empirical max-length-hit rate, and empirical accuracy.
  - This is the missing bridge between "easy to predict" and "actually useful for catching bad rollouts."
  - If the cheap prompt-length or joint metadata baseline wins this test, keep it as a valid operational screen and state clearly that the gain is metadata-driven rather than activation-driven.
- Rebuild the compact five-dataset summary once the metadata baselines above exist, so future head choices are judged against the strongest non-activation model rather than against prompt length alone.

## Fixed Experimental Surface

- Keep the predictor input to prompt-prefill activations only.
- Stay in-distribution and prompt-disjoint.
- Keep one selected layer or per-layer independent probes with late aggregation.
- Keep the decode policy fixed at `temperature = 0.2` while settling the objective question.

## Measurement And Reporting Gaps

- The explicit cross-dataset `majority_s_0.5` table now exists under `outputs/prompt_majority05_cross_dataset_rebuild_20260325/`; future replies should cite that table directly instead of falling back to `AIME`-only anecdotes.
- That rebuilt table answers only the binary prompt-length question. It does **not** answer the collaborator's actual degeneracy-screen question yet; the bucket test on empirical loop rate, max-length-hit rate, and accuracy is still missing.
- The current five-dataset continuous-head table still uses raw prompt-length association rather than a trained metadata baseline.
- The current saved summaries still do not log the joint `prompt_length + effective_budget` baseline.
- Do not describe the current continuous-head prompt-length rows as a metadata-only predictor; they are descriptive correlations only.
- Older thread notes used "rank correlation" as shorthand. Future writeups should say `Spearman rank correlation` explicitly and always name the target being ranked.
- Older notes also used "prompt-length baseline" too loosely. Future writeups should say whether this means a train-fit 1D scorer or only a raw held-out association statistic.

## Known Data Gaps

- The original `LiveCodeBench` job crashed after grading and before writing its final JSON. Replay-based repair did not recover `avg_first_loop_prefix_length` exactly. That metric remains `null` in the recovered capped bundle. A fresh rerun would be required if exact prefix-length telemetry is still needed.

## Open Implementation Issues

- This workspace still does not have a local Torch runtime / project virtualenv. Code paths are syntax-checked locally; real Torch-backed build/train execution requires the remote pilot window.

## Conditional Next Step

- If the metadata-baseline pass and top-risk-bucket comparison still leave the binary-head choice unsettled, reopen direct `p_cap` next rather than another threshold sweep or `loop_budget_share`.

## Open Review Surfaces

- Upstream PR #7 (`Zhi0467/cot-loop`, branch `task/1773870804-prompt-profile-probe`) — prompt-profile implementation.
- Upstream PR #6 (`Zhi0467/cot-loop`, branch `task/1773451376-common-policy-refresh`) — common-policy rollout bundle.
