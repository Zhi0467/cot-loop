# Thread Reset - 2026-03-25 Fundamentals Handoff

Last updated: 2026-03-25 06:10 UTC

## Why This Note Exists

- The last Slack thread drifted away from the collaborator's object.
- The next thread should start from a proved-vs-unproved ledger plus an exact work order, not from another abstract head recommendation.

## What The Collaborator Corrected

These points summarize the collaborator's last stretch of messages from `2026-03-24 23:59 UTC` through `2026-03-25 06:07 UTC`.

- Define every "baseline" precisely. "Prompt length alone" is not acceptable shorthand unless the actual algorithm is stated.
- Do not say a metadata baseline "works" when only raw `Spearman(prompt_length, target)` exists. Correlation tables are not trained predictors.
- Do not answer the wrong object. The urgent question is whether prompt length and `majority_s_0.5` actually work across datasets for catching degenerate rollouts.
- Do not lean on one narrow slice such as `AIME` plus `PR-AUC` alone. Cross-dataset comparison against learned probes is the real requirement.
- Keep the project objective explicit: catching degenerate rollouts matters more than defending activation-specific lift. A cheap metadata-driven screen is acceptable if it really isolates bad prompts.
- Metric names must stay explicit and target-specific. Say `Spearman rank correlation`, `top-20% capture`, and name the target whose mass is being captured.
- Stop moving back to "which head should ship?" before the fundamentals are settled.
- Keep the roadmap / backlog current so the next thread can start from the real unfinished work.
- Do not send duplicate or overlapping final replies in Slack.

## Proven Today

- Binary `majority_s_0.5` already has a real prompt-length-only held-out baseline:
  - feature = `prompt_token_count` only
  - fit orientation and threshold on train prompts
  - freeze the rule
  - evaluate on held-out prompts
- That binary prompt-length result is dataset-conditional, not global:
  - `GPQA`: weak (`PR-AUC 0.0817`)
  - `AIME`: strong (`PR-AUC 0.8976`)
- For the current continuous-head table (`p_loop`, `mean_relative_length`, `loop_budget_share`), the saved prompt-length rows are still only raw held-out association, not trained metadata-only predictors.
- The rollout-statistics bundle already proves the degenerate regime is real on hard datasets: loops are longer, less accurate, and tightly tied to max-length hits.

## Not Proven Yet

- Whether prompt length "works" as a cross-dataset claim beyond the narrow binary surfaces already measured.
- Whether `majority_s_0.5` actually catches meaningfully degenerate prompts across datasets rather than mostly catching prompt geometry / frequent long-rollout behavior.
- Whether activation probes beat the strongest metadata-only baselines on the continuous heads once those baselines are trained for real.

## Exact Next Work Order

Hold fixed:

- prompt-prefill activations only
- in-distribution prompt-disjoint evaluation
- one selected layer or per-layer independent probes with late aggregation
- fixed decode policy with `temperature = 0.2`

Then do the next work in this order:

1. Rebuild the cross-dataset `majority_s_0.5` scoreboard in explicit side-by-side form.
   - For every finished dataset, report the prompt-length-only 1D rule beside the learned last-layer / ensemble probes.
   - Use more than `PR-AUC`: include `ROC-AUC`, `macro-F1`, positive precision / recall, and prevalence.
   - This is the direct answer to "does prompt length actually work?" on the only surface where a real prompt-length model already exists.
2. Run the first true metadata-only continuous-head pass.
   - Fit `prompt_length` only, `effective_budget` only, and `prompt_length + effective_budget` on the same held-out splits for `p_loop` and `mean_relative_length`.
   - Do not reuse raw `Spearman(prompt_length, target)` as if it were a trained baseline.
3. Run the decisive prompt-level usefulness test on held-out prompts.
   - Compare the top predicted-risk `20%` prompts under `majority_s_0.5`, `p_loop`, `mean_relative_length`, and the strongest metadata baseline.
   - Measure empirical loop rate, empirical max-length-hit rate, and empirical accuracy inside those buckets.
4. Make the operational claim only after Step 3.
   - If the cheap metadata baseline wins, say plainly that the useful screen is metadata-driven.
   - If an activation-based score wins, then claim activation lift with the correct control.
5. Only if the binary decision still remains unsettled after the steps above, reopen direct `p_cap`.

## Correct New-Thread Starting Point

The next thread should start from this question:

> Under the fixed in-distribution prompt-profile setup, is `majority_s_0.5` actually a good cross-dataset screen for degenerate rollouts, and if so is its usefulness coming from activations or from cheap metadata alone?
