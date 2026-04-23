# Prompt-Profile Risk-Screen Decision

Last updated: 2026-03-30 10:55 UTC

## Correction

- The previous revision used the held-out `top 20%` bucket test to choose the training objective.
- That was the wrong selector.
- The bucket test is a downstream screen diagnostic.
- The objective choice should be made from held-out predictability on the target itself.

## Question

Under the fixed prompt-profile setup, which prompt-level targets are actually most learnable from prompt-prefill activations on held-out prompts?

That splits into two separate questions:

- which binary label is most learnable?
- which continuous target is most learnable?

## Artifacts

- Risk-control analysis bundle: `outputs/weeks/2026-W14/prompt_profile_risk_controls_20260330/`
- Cross-dataset summary JSON: `outputs/weeks/2026-W14/prompt_profile_risk_controls_20260330/cross_dataset_summary.json`
- Metadata-baseline metric table: `outputs/weeks/2026-W14/prompt_profile_risk_controls_20260330/metadata_baseline_metrics.csv`
- Majority-label rebuild table: `outputs/weeks/2026-W13/prompt_majority05_cross_dataset_rebuild_20260325/prompt_majority05_cross_dataset_rebuild_20260325_aggregate.csv`

## What Was Compared

- Saved datasets: `GPQA`, `AIME`, `MATH-500`, `MMLU-Pro`, `LiveCodeBench`
- Existing activation probes:
  - binary `majority_s_0.5`
  - continuous `p_loop`
  - continuous `mean_relative_length`
- Existing readouts:
  - last-layer MLP
  - per-layer ensemble MLP
- New metadata-only controls trained in this pass:
  - logistic regression for `p_loop`
  - linear regression for `mean_relative_length`
  - features = `prompt_length`, `effective_budget`, and `prompt_length + effective_budget`

## Held-Out Predictability Ledger

### Continuous Targets

Best held-out `Spearman` by dataset:

| Dataset | Best `p_loop` activation score | Best `mean_relative_length` activation score | Best `p_loop` metadata score | Best `mean_relative_length` metadata score |
| --- | ---: | ---: | ---: | ---: |
| `AIME` | `0.475` | `0.853` | `0.459` | `0.676` |
| `GPQA` | `0.321` | `0.649` | `-0.115` | `0.116` |
| `MATH-500` | `0.246` | `0.716` | `0.119` | `0.461` |
| `MMLU-Pro` | `0.339` | `0.303` | `0.094` | `0.393` |
| `LiveCodeBench` | `0.478` | `0.785` | `0.191` | `0.405` |

Read:

- `mean_relative_length` is the stronger regression target on `4 / 5` datasets.
- Average best held-out `Spearman` is about `0.661` for `mean_relative_length` versus `0.372` for `p_loop`.
- Average activation-over-metadata lift is also slightly larger for `mean_relative_length` (`~0.251`) than for `p_loop` (`~0.222`).
- `p_loop` does still beat its metadata baseline on all five datasets, so there is real activation signal, but it is not the strongest current regression target.

### Binary Label

Best held-out `PR-AUC` for `majority_s_0.5` against the prompt-length-only control:

| Dataset | Prompt-length-only `PR-AUC` | Best `majority_s_0.5` `PR-AUC` |
| --- | ---: | ---: |
| `GPQA` | `0.082` | `0.667` |
| `AIME` | `0.806` | `0.937` |
| `MATH-500` | `0.103` | `0.216` |
| `MMLU-Pro` | `0.110` | `0.142` |
| `LiveCodeBench` | `0.576` | `0.751` |

Read:

- `majority_s_0.5` is the only mature binary label surface in this saved bundle.
- It is learnable enough to use as the current binary control.
- But it is clearly geometry-shaped on some slices, especially `AIME`.

## What The Old Bucket Test Still Means

The old bucket test remains true as a downstream statement:

- a `p_loop` score still isolates loop-heavy prompts better than the other saved scores in that fixed `top 20%` comparison
- on the four datasets with prompt-level accuracy, that same bucket is also the lowest-accuracy bucket

But that should now be read narrowly:

- it says something about downstream concentration after scoring
- it does **not** say that `p_loop` is the best label to train

## Decision

By the predictability-first criterion:

- Best regression target: `mean_relative_length`
- Best binary label: `majority_s_0.5`
- `p_loop`: keep as the most loop-specific downstream target candidate, but do not claim it is the main training objective from the current saved evidence
- `p_cap`: still closed

## Interpretation

- `mean_relative_length` is currently the cleanest thing the prefill probes can regress well.
- `majority_s_0.5` is currently the cleanest finished binary label surface, though partly prompt-length-shaped.
- `p_loop` is semantically closer to looping, but the current models do not predict it as well as `mean_relative_length`.

The likely driver is target density rather than the bucket result:

- `p_loop` target means on held-out prompts are only `0.104` (`AIME`), `0.138` (`GPQA`), `0.025` (`MATH-500`), `0.036` (`MMLU-Pro`), and `0.148` (`LiveCodeBench`)
- `mean_relative_length` is dense and uses every rollout

That density explanation is an inference from the saved metrics, not a new experiment.

## Caveats

- This still does not collapse binary and regression into one single winner. That choice has to be made explicitly.
- This is still retrospective on saved bundles, not a fresh prospective run.
- `LiveCodeBench` still lacks prompt-level accuracy in the recovered projection artifact.
- The earlier bucket result is still useful operationally, but it cannot be used as the objective selector.

## Exact Next Run

- Decide whether the product wants a binary label or a continuous target.
- If regression is the live path:
  - run a fresh prompt-disjoint `mean_relative_length` versus `p_loop` comparison
  - select by held-out regression quality on the target itself plus metadata lift
- If binary classification is the live path:
  - keep `majority_s_0.5` as the current binary control
  - only reopen a direct loop-derived binary label if it wins on held-out classification quality
- If the goal is specifically to rescue `p_loop`, judge it by held-out `p_loop` fit, not by bucket enrichment
