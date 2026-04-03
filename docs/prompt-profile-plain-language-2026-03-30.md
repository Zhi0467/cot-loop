# Prompt-Profile Objective In Plain Words

Last updated: 2026-03-30 10:55 UTC

## One Correction First

- The previous version used the wrong selector.
- The `top 20%` bucket test answered: "after training, which score happens to concentrate loop-heavy prompts best?"
- Wangzhi's actual question is different: "which prompt-level label or regression target can we predict well from prompt-prefill activations on held-out prompts?"
- Those are not the same question.

## One-Sentence Answer

If the selector is held-out predictability rather than downstream bucket behavior:

- best regression target right now: `mean_relative_length`
- best binary label right now: `majority_s_0.5`
- `p_loop` is still the most loop-specific target, but the current saved runs do not show that it is the best-predicted target

## What The Project Is Actually Trying To Do

- Fix one `(model, decode policy)` pair.
- For each prompt, define one prompt-level label or regression target from repeated rollouts.
- Train on prompt-prefill activations only.
- Choose the target that the current probes can predict well on held-out prompts.

Only after that does it make sense to ask whether the chosen target is useful for a downstream screen.

## Definitions

### `p_loop`

- Plain meaning: for one prompt, what fraction of repeated rollouts enter the loop regime.
- Example: if `3` of `10` rollouts loop, then `p_loop = 0.3`.

### `p_cap`

- Plain meaning: for one prompt, what fraction of repeated rollouts hit the hard max-length cap.

### `mean_relative_length`

- Plain meaning: for one prompt, what fraction of the available generation budget gets used on average.
- This is a dense continuous target. Every rollout contributes.

### `majority_s_0.5`

- Plain meaning: a binary label that is `1` if more than half of repeated rollouts use at least half of the available budget.
- This is better read as "often long" than as "often loops."

### `effective_budget`

- Plain meaning: the real generation budget left after prompt length and context limits are taken into account.

### `flagged top-risk bucket`

- This is only an old evaluation slice.
- For one trained score on held-out prompts, take the `20%` with the largest predicted score and inspect what actually happened there.
- It is not the project goal and it is not the target-selection rule.

### `loop-rate enrichment`

- Definition:
  `loop-rate enrichment = (loop rate inside the flagged bucket) / (overall loop rate on that test split)`
- Example: `2.0x` means the flagged bucket loops twice as often as the dataset average.

## What Was Trained

This pass used the finished saved prompt-profile bundles for:

- `GPQA`
- `AIME`
- `MATH-500`
- `MMLU-Pro`
- `LiveCodeBench`

For each dataset, the saved surface already contained activation probes trained from prompt-prefill activations for:

- `majority_s_0.5`
- `p_loop`
- `mean_relative_length`

and two readouts:

- last-layer MLP
- per-layer ensemble MLP

This pass also trained the missing metadata-only controls:

- logistic regression for `p_loop`
- linear regression for `mean_relative_length`
- features = `prompt_length`, `effective_budget`, and `prompt_length + effective_budget`

The split stayed prompt-disjoint:

- train prompts and test prompts are different prompts
- repeated rollouts from one prompt never leak across the split

## What Should Pick The Target

The selector should be the held-out fit on the target itself.

- For a binary label, that means classification quality on that label itself.
  - Here the saved surface gives `PR-AUC`, `ROC-AUC`, precision, recall, and related metrics.
- For a continuous target, that means regression quality on that target itself.
  - Here the saved surface gives `Spearman`, `MAE`, and `RMSE`.

The bucket test is a downstream diagnostic. It should not be the reason we decide which label to train.

## What The Saved Bundle Actually Says

### Continuous Targets

Best held-out `Spearman` from the existing activation probes:

| Dataset | Best `p_loop` `Spearman` | Best `mean_relative_length` `Spearman` | Best metadata baseline for `p_loop` | Best metadata baseline for `mean_relative_length` |
| --- | ---: | ---: | ---: | ---: |
| `AIME` | `0.475` | `0.853` | `0.459` | `0.676` |
| `GPQA` | `0.321` | `0.649` | `-0.115` | `0.116` |
| `MATH-500` | `0.246` | `0.716` | `0.119` | `0.461` |
| `MMLU-Pro` | `0.339` | `0.303` | `0.094` | `0.393` |
| `LiveCodeBench` | `0.478` | `0.785` | `0.191` | `0.405` |

Readout:

- `mean_relative_length` is the easier regression target on `4 / 5` datasets.
- Average best held-out `Spearman` is about `0.661` for `mean_relative_length` versus `0.372` for `p_loop`.
- `mean_relative_length` also beats its best metadata baseline on `4 / 5` datasets.
- `p_loop` does beat its metadata baseline on all `5 / 5` datasets, so the activations are not useless there, but the absolute fit is still clearly weaker.

So if the question is "which continuous prompt-level target can we currently regress best from prefill activations?", the answer is `mean_relative_length`, not `p_loop`.

### Binary Label

For the saved binary surface, the mature label is `majority_s_0.5`.

| Dataset | Prompt-length-only `PR-AUC` | Best `majority_s_0.5` `PR-AUC` |
| --- | ---: | ---: |
| `GPQA` | `0.082` | `0.667` |
| `AIME` | `0.806` | `0.937` |
| `MATH-500` | `0.103` | `0.216` |
| `MMLU-Pro` | `0.110` | `0.142` |
| `LiveCodeBench` | `0.576` | `0.751` |

Readout:

- If we need a binary label today, `majority_s_0.5` is the current best-supported one because it is the only one here with a finished held-out classification surface.
- But it is not a pure loop label.
- On `AIME` especially, it is already strongly prompt-length-shaped, so some of that predictability is coming from prompt geometry rather than a cleaner loop-specific signal.

## What This Means For `p_loop`

`p_loop` is still important, but the claim needs to be narrower.

- It is the most direct loop-frequency target in this bundle.
- The old bucket test still shows that a `p_loop` score can isolate loop-heavy prompts downstream.
- But by the criterion Wangzhi asked for, that is not enough.

The current saved runs do **not** show that `p_loop` is the target we can predict best.

The most obvious reason in the saved tables is target density:

- `p_loop` is sparse on these held-out splits.
  - held-out target means are only `0.104` on `AIME`, `0.138` on `GPQA`, `0.025` on `MATH-500`, `0.036` on `MMLU-Pro`, and `0.148` on `LiveCodeBench`
- `mean_relative_length` is dense and uses every rollout.

That density difference is consistent with why the current MLPs regress `mean_relative_length` more cleanly. This is an inference from the saved results, not a new causal proof.

## So What Is The Right Objective?

There are now two honest answers, depending on output type:

- If the project wants one continuous target to regress well from prefill activations, use `mean_relative_length`.
- If the project wants one binary label to classify with the current saved surface, use `majority_s_0.5`.

What I should **not** say anymore is "train `p_loop` because it wins loop-rate enrichment." That used the wrong rule.

The correct statement is:

- `p_loop` is still the best loop-specific downstream diagnostic in the old bucket comparison
- `p_loop` is **not yet proved** to be the best training target by held-out predictability

## What This Resolves From The Last Thread

This fixes the main logic error in the previous write-up:

- the old note used a downstream concentration metric to choose the training target
- the corrected note separates target predictability from downstream usefulness
- the saved bundle now supports a clean proved-versus-unproved ledger

Proved on the current saved bundle:

- `mean_relative_length` is the strongest regression target by held-out fit
- `majority_s_0.5` is the strongest existing binary label surface
- `p_loop` still contains real activation signal beyond metadata

Not proved on the current saved bundle:

- that `p_loop` should be the main training objective
- that the project should collapse binary and regression into one final answer without first choosing output type
- that the old bucket result should select the label

## What Still Does Not Resolve

Three things are still open:

1. This is still retrospective on saved bundles rather than one fresh prospective run.
2. `LiveCodeBench` still lacks prompt-level accuracy in the recovered projection artifact, so the accuracy-side comparison is still incomplete there.
3. If the real goal is specifically loop prediction, then the current answer is not "use `p_loop`." The current answer is "we have not yet shown that `p_loop` is the most learnable target."

## Exact Next Run

- Decide the output type first.
  - If regression is the product, compare `mean_relative_length` and `p_loop` prospectively and choose by held-out regression quality plus metadata lift.
  - If binary classification is the product, keep `majority_s_0.5` as the current binary control and only reopen a direct loop-derived binary label if it actually wins on held-out classification quality.
- If the goal is specifically to rescue `p_loop`, freeze the rule up front:
  - judge it by held-out fit on `p_loop` itself
  - keep the metadata baselines mandatory
  - do not let bucket enrichment choose the objective again
