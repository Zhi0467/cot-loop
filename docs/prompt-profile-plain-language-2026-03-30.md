# Prompt-Profile Objective In Plain Words

Last updated: 2026-03-30 10:15 UTC

## One Correction First

- The project goal is **not** "rank prompts."
- The project goal is: choose the right prompt-level label or regression target to predict from prompt-prefill activations under one fixed model and one fixed decode policy.
- I used one common `top 20%` bucket diagnostic only to compare candidate targets on the same footing. That diagnostic is not the goal of the project.

## One-Sentence Answer

On the current saved five-dataset bundle, the best main target is `p_loop`: for each prompt, regress the fraction of rollouts that enter the loop regime.

## What The Project Is Actually Trying To Do

- Fix one `(model, decode policy)` pair.
- For each prompt, define one prompt-level target from repeated rollouts.
- Ask which target is the best thing to learn from prompt-prefill activations.
- "Best" here means: the learned score should track the bad behavior we actually care about, not just prompt geometry or budget usage.

So the project question is a target-choice question:

- should we train on `majority_s_0.5`?
- should we regress `p_loop`?
- should we regress `mean_relative_length`?
- should we regress `p_cap`?

The answer now is `p_loop`.

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

### `flagged top-risk bucket`

- This phrase only refers to an evaluation slice.
- For one trained model on one held-out test split, take the `20%` of prompts with the largest predicted badness score.
- Then check what really happened on those prompts.
- It is not a new project goal and it is not a new label. It is just one way to compare different candidate targets with the same bucket size.

### `loop-rate enrichment`

- Definition:
  `loop-rate enrichment = (loop rate inside the flagged bucket) / (overall loop rate on the test split)`
- Example: `2.0x` means the flagged bucket has twice the average loop rate. `1.0x` means no concentration. Below `1.0x` means the bucket is worse than the dataset average.

### Why use `loop-rate enrichment` at all?

- Different datasets have very different base loop rates, so raw loop rate is not directly comparable across `AIME`, `GPQA`, `MATH-500`, `MMLU-Pro`, and `LiveCodeBench`.
- The candidate targets also live on different scales:
  - `majority_s_0.5` is binary;
  - `p_loop` and `p_cap` are probabilities;
  - `mean_relative_length` is a continuous budget-use number.
- `loop-rate enrichment` gives one simple common check: when a target says "these prompts look bad," do those prompts actually loop more often than usual?
- So enrichment is an evaluation tool for target choice, not the definition of success by itself.

## What Was Missing From The Last Thread

The last thread was still missing two decisive pieces:

1. Real metadata-only baselines for the continuous heads.
   - Not just raw `Spearman(prompt_length, target)`.
   - Actual trained models using `prompt_length`, `effective_budget`, or both.
2. A common held-out bucket test.
   - On held-out prompts, take the `20%` with the largest predicted badness score.
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

- existing activation probes already trained from prompt-prefill activations:
  - `majority_s_0.5`
  - `p_loop`
  - `mean_relative_length`
- existing activation readouts:
  - last-layer MLP
  - per-layer ensemble MLP
- new metadata-only controls trained in this pass:
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
5. Take the `20%` of prompts with the largest predicted badness score.
6. Measure what actually happened inside that bucket:
   - true loop rate;
   - true cap-hit rate;
   - true accuracy when prompt-level correctness exists.

This bucket test is only one evaluation device. I used it because it lets different target types share one simple held-out check without choosing a custom threshold for every target.

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

On the four datasets where prompt-level accuracy is available (`GPQA`, `AIME`, `MATH-500`, `MMLU-Pro`), the `p_loop` bucket is also the lowest-accuracy bucket every time.

### Why That Matters

If the project goal is "choose the prompt-level target that best matches real looping behavior," then `p_loop` is the most aligned target:

- it directly asks how often the prompt loops;
- it consistently concentrates loop-heavy prompts;
- it also concentrates low-accuracy prompts where accuracy is available.

That is why the evidence now points to `p_loop`, not because the project quietly changed into a ranking task.

## Why The Other Objectives Lost

### Why not `majority_s_0.5`

- It is a coarse thresholded label.
- It mainly asks whether the prompt is often long, not whether it often loops.
- Prompt length alone already does very well on this label in some slices, especially `AIME`.

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

The main reason is alignment between the target and the thing we actually want to predict:

- If you train `majority_s_0.5`, you teach the model to predict a coarse "often long" label.
- If you train `mean_relative_length`, you teach the model to predict budget usage.
- If you train `p_cap`, you teach the model to predict only the narrow cap-hit subset.
- If you train `p_loop`, you teach the model to predict the event we actually want to screen for: entering the loop regime.

So the winning result is not "mysteriously better training." The target itself matches the failure mode better, and that survives the held-out comparison.

## What This Resolves From The Last Thread

This pass resolves the main open issue from the last thread:

- the decision is no longer resting on one dataset anecdote;
- it is no longer resting on raw prompt-length correlation for the continuous heads;
- it is no longer missing the metadata-only baseline pass;
- it is no longer missing one common held-out comparison slice.

So the target choice is now materially better supported:

- main objective: `p_loop`
- secondary utility head: `mean_relative_length`
- control / cheap auxiliary screen: `majority_s_0.5`
- do not reopen direct `p_cap` now

## What It Does Not Resolve Yet

Three limits still matter:

1. This is retrospective on saved bundles, not a fresh prospective run.
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
- Train `mean_relative_length` only if a secondary utility or budget score is still wanted.
- Keep `majority_s_0.5` in the evaluation bundle as a control, not as the main target.
