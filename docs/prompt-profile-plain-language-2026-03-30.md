# Prompt-Profile Objective In Plain Words

Last updated: 2026-03-30 09:14 UTC

## One-Sentence Answer

The project wants a predictor that looks only at prompt-prefill activations and flags prompts that are likely to make the model enter a bad rollout regime under one fixed model and one fixed decode policy. On the current saved five-dataset bundle, the best main target for that job is `p_loop`.

## What The Project Is Actually Trying To Do

- We are not trying to predict whether a prompt is "inherently loopy" in the abstract.
- We are trying to predict what usually happens for one fixed `(model, prompt, decode policy)` tuple before generation starts.
- The practical use is prompt screening: rank prompts by risk, then inspect or avoid the worst ones.
- "Bad" here mainly means:
  - the rollout falls into the loop regime;
  - the rollout burns a lot of budget and may hit the cap;
  - accuracy drops sharply inside that risky slice.

So the project goal is: use prompt-prefill activations to find prompts that are likely to go bad later, not just prompts that are long.

## Definitions

### `p_loop`

- Plain meaning: for one prompt, how often repeated rollouts loop.
- Example: if `3` out of `10` rollouts loop, then `p_loop = 0.3`.
- Why it matters: this is the most direct prompt-level version of the failure mode we care about.

### `p_cap`

- Plain meaning: for one prompt, how often repeated rollouts hit the hard max-length cap.
- Why it matters: cap hits are a sharp bad sign, but they are narrower than loops because many loops stop before the cap.

### `mean_relative_length`

- Plain meaning: on average, what fraction of the available generation budget the prompt consumes.
- Example: if a prompt usually uses about half its allowed generation budget, this number is around `0.5`.
- Why it matters: it is a useful budget or difficulty proxy, but it mixes good long reasoning and bad long reasoning.

### `majority_s_0.5`

- Plain meaning: a binary label that is `1` if more than half of the repeated rollouts use at least half of the available budget.
- This is not a direct loop label.
- It is better read as "this prompt is often long" rather than "this prompt often loops."

### `effective_budget`

- Plain meaning: the actual generation budget available after prompt length and context-limit constraints are taken into account.

### `loop-rate enrichment`

- Definition:
  `loop-rate enrichment = (loop rate inside the top predicted-risk bucket) / (overall loop rate on the test split)`
- Example: `2.0x` means the flagged bucket has twice the average loop rate.
- This is important because the downstream use is ranking prompts and then looking at the worst slice.

## What Was Missing From The Last Thread

The last thread was still missing two decisive pieces:

1. Real metadata-only baselines for the continuous heads.
   - Not just raw `Spearman(prompt_length, target)`.
   - Actual trained models using `prompt_length`, `effective_budget`, or both.
2. An operational bucket test.
   - Rank held-out prompts by predicted risk.
   - Keep the top `20%`.
   - Measure the actual loop rate, actual cap-hit rate, and actual accuracy inside that bucket.

Without those two pieces, the old thread could not cleanly answer which objective should actually be trained.

## What Was Trained And Compared

This pass used the finished saved prompt-profile bundles for five datasets:

- `GPQA`
- `AIME`
- `MATH-500`
- `MMLU-Pro`
- `LiveCodeBench`

For each dataset, the comparison surface contained:

- Existing activation probes already trained from prompt-prefill activations:
  - `majority_s_0.5`
  - `p_loop`
  - `mean_relative_length`
- Existing activation readouts:
  - last-layer MLP
  - per-layer ensemble MLP
- New metadata-only controls trained in this pass:
  - logistic regression for `p_loop`
  - linear regression for `mean_relative_length`
  - features = `prompt_length`, `effective_budget`, and `prompt_length + effective_budget`

The split stayed prompt-disjoint:

- train on one set of prompts;
- test on different prompts;
- never let the same prompt leak across train and test through repeated rollouts.

## How The Comparison Worked

For each prompt:

1. Run repeated rollouts under one fixed decode policy.
2. Turn those rollouts into prompt-level target numbers such as `p_loop` and `mean_relative_length`.
3. Train on train prompts only.
4. Score held-out test prompts.
5. Sort test prompts by predicted risk.
6. Keep the top `20%` highest-risk prompts.
7. Measure what actually happened inside that bucket:
   - true loop rate;
   - true cap-hit rate;
   - true accuracy when prompt-level correctness exists.

This bucket test is the key operational check. It answers "if we really use this score to flag bad prompts, does it isolate the bad slice?" That is more important than only reporting correlation or loss.

## Why `p_loop` Wins

### Result Summary

| Dataset | `majority_s_0.5` loop enrichment | `p_loop` loop enrichment | Strongest metadata baseline |
| --- | ---: | ---: | ---: |
| `GPQA` | `1.82x` | `1.82x` | `0.91x` |
| `AIME` | `0.80x` | `3.20x` | `0.80x` |
| `MATH-500` | `2.50x` | `3.50x` | `1.50x` |
| `MMLU-Pro` | `2.61x` | `3.48x` | `1.74x` |
| `LiveCodeBench` | `1.63x` | `2.32x` | `1.26x` |

Average loop-rate enrichment across datasets:

- `p_loop`: `2.86x`
- `majority_s_0.5`: `1.87x`
- `mean_relative_length`: `1.77x`
- strongest metadata baseline: `1.24x`

On the four datasets where prompt-level accuracy is available (`GPQA`, `AIME`, `MATH-500`, `MMLU-Pro`), the `p_loop` top-risk bucket is also the lowest-accuracy bucket every time.

### Why That Matters

If the operational goal is "find prompts that actually go bad," then `p_loop` is the target most aligned with that job:

- it directly asks how often the prompt loops;
- it consistently concentrates loop-heavy prompts;
- it also concentrates low-accuracy prompts where accuracy is available.

That is exactly the screening behavior we wanted.

## Why The Other Objectives Lost

### Why not `majority_s_0.5`

- It is a coarse thresholded label.
- It mainly asks whether the prompt is often long, not whether it often loops.
- Prompt length alone already does very well on this label in some slices, especially `AIME`.
- So it still has signal, but it is too geometry-heavy to be the main headline objective.

This is why it should stay as a control or cheap auxiliary screen, not the main training target.

### Why not `mean_relative_length`

- It is a useful dense target.
- It ranks budget usage well.
- But it mixes several behaviors together:
  - correct long reasoning;
  - wrong long reasoning;
  - looping.

So it is good as a secondary utility or budget head, but it is not the cleanest main bad-prompt screen.

### Why not `p_cap`

- It is too narrow.
- Many bad looped rollouts do not actually hit the cap.
- So training only on cap hits would miss a lot of the failure mass we care about.

## Why This Training Objective Leads To The Best Result

The main reason is alignment between the target and the downstream decision:

- If you train `majority_s_0.5`, you teach the model to predict a coarse "often long" label.
- If you train `mean_relative_length`, you teach the model to predict budget usage.
- If you train `p_cap`, you teach the model to predict only the narrow cap-hit subset.
- If you train `p_loop`, you teach the model to predict the event we actually want to screen for: entering the loop regime.

The winning result is therefore not "mysteriously better training." It is that the objective itself matches the operational question better, and that advantage survives the held-out bucket test.

## What This Resolves From The Last Thread

This pass resolves the main open issue from the last thread:

- the decision is no longer resting on one dataset anecdote;
- it is no longer resting on raw prompt-length correlation for the continuous heads;
- it is no longer missing the operational bucket test.

So the target choice is now materially better supported:

- main objective: `p_loop`
- secondary utility head: `mean_relative_length`
- control / cheap auxiliary screen: `majority_s_0.5`
- do not reopen direct `p_cap` now

## What It Does Not Resolve Yet

Three limits still matter:

1. This is retrospective on saved bundles, not a fresh prospective deployment run.
2. `LiveCodeBench` still lacks prompt-level accuracy in the recovered projection artifact, so the "lowest-accuracy bucket" claim is `4 / 5` datasets rather than `5 / 5`.
3. Small datasets, especially `AIME`, can still move noticeably when only a few prompts change buckets.

So the objective choice is settled much better than before, but the next confirmation should still be prospective.

## Exact Recommended Next Run

- Freeze the architecture, feature view, and checkpoint-selection rule up front.
- Train one fresh prompt-disjoint `p_loop` predictor on a representative hard slice.
- Keep the metadata baselines mandatory:
  - `prompt_length`
  - `effective_budget`
  - `prompt_length + effective_budget`
- Train `mean_relative_length` only if we also want a secondary utility or budget score.
- Keep `majority_s_0.5` in the evaluation bundle as a control, not as the main target.

## Bottom Line

The plainest summary is:

- the project goal is to flag prompts that make the model go bad before generation starts;
- the missing baselines and bucket test are now done on the saved five-dataset bundle;
- `p_loop` is the objective most aligned with that goal and the one that wins the actual held-out risk-bucket test;
- `mean_relative_length` is still useful, but as a secondary budget head;
- `majority_s_0.5` still has signal, but it is too close to a geometry-heavy "often long" control to be the main objective.
