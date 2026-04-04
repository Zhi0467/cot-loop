# Prompt-Profile Full Train Results

Last updated: 2026-04-04 00:17 UTC

## Question

What does the first locked full-train pass say about the current two-head surface under the fixed saved prompt-profile contract?

- regression head: `mean_relative_length`
- binary head: `majority_s_0.5`
- feature surface: prompt-prefill activations only
- views compared: layerwise `ensemble` vs `last_layer`

## Setup

- Runtime object: Slurm job `2043` on `2` GPUs, completed at `2026-04-04 00:10 UTC`.
- Data object: reused the saved March prompt-profile archives; this pass did not reroll prompts or rebuild labels from scratch.
- Evaluation object: use the standardized ledger from `scripts/summarize_prompt_profile_full_train.py`, with `best_loss` as the main checkpoint and `best_rank` left diagnostic-only.
- Important mechanism split:
  - for regression `mean_relative_length`, `ensemble` means one MLP per layer with `mean_prob` aggregation;
  - for binary `majority_s_0.5`, `ensemble` means one MLP per layer with `vote_fraction` aggregation over the per-layer 0/1 predictions.
- Metadata baselines below mean train-fit held-out scorers, not raw correlations.
- On this fixed run surface, `effective_budget` is constant at `30000`, so the joint regression control collapses to the prompt-length-only baseline.

Copied ledger for this run:
- `outputs/prompt_profile_full_train_locked_pair_20260404/summary/`
- `outputs/prompt_profile_full_train_locked_pair_20260404/logs/`
- `outputs/prompt_profile_full_train_locked_pair_20260404/regression_summary.csv`
- `outputs/prompt_profile_full_train_locked_pair_20260404/binary_summary.csv`

## Main Results

### Regression `mean_relative_length`

The activation ensemble does beat the last-layer control on all five datasets in `Spearman rank correlation`, but that is not the same as beating metadata.

| Dataset | Ensemble `Spearman` | Last-layer `Spearman` | Prompt-length baseline `Spearman` | Ensemble `RMSE` | Last-layer `RMSE` | Prompt-length baseline `RMSE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.419 +/- 0.045` | `0.175 +/- 0.225` | `0.116` | `0.160` | `0.164` | `0.165` |
| `AIME` | `0.639 +/- 0.090` | `0.483 +/- 0.138` | `0.676` | `0.183` | `0.189` | `0.176` |
| `MATH-500` | `0.351 +/- 0.199` | `0.158 +/- 0.426` | `0.461` | `0.162` | `0.169` | `0.154` |
| `MMLU-Pro` | `0.348 +/- 0.034` | `-0.019 +/- 0.293` | `0.393` | `0.133` | `0.138` | `0.126` |
| `LiveCodeBench` | `0.805 +/- 0.002` | `0.784 +/- 0.006` | `0.405` | `0.248` | `0.242` | `0.279` |

Clean read:
- The activation ensemble is clearly better than `last_layer` on the regression head.
- The activation ensemble only beats the trained prompt-length baseline on `GPQA` and `LiveCodeBench`.
- `AIME`, `MATH-500`, and `MMLU-Pro` remain metadata-dominated on this regression target.

### Binary `majority_s_0.5`

The binary head is the stronger part of the locked pair. The main ranking metric here is `PR-AUC`, especially on the very low-prevalence datasets.

| Dataset | Ensemble `PR-AUC` | Last-layer `PR-AUC` | Prompt-length baseline `PR-AUC` | Ensemble macro-F1 | Last-layer macro-F1 | Prompt-length baseline macro-F1 | Test prevalence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.420 +/- 0.029` | `0.230 +/- 0.278` | `0.066` | `0.382` | `0.414` | `0.286` | `0.050` |
| `AIME` | `0.922 +/- 0.024` | `0.904 +/- 0.028` | `0.898` | `0.798` | `0.391` | `0.667` | `0.583` |
| `MATH-500` | `0.135 +/- 0.005` | `0.151 +/- 0.071` | `0.100` | `0.444` | `0.468` | `0.458` | `0.040` |
| `MMLU-Pro` | `0.184 +/- 0.037` | `0.056 +/- 0.012` | `0.110` | `0.562` | `0.357` | `0.397` | `0.013` |
| `LiveCodeBench` | `0.711 +/- 0.036` | `0.712 +/- 0.023` | `0.576` | `0.747` | `0.714` | `0.646` | `0.338` |

Clean read:
- Ensemble `PR-AUC` beats the prompt-length baseline on all five datasets.
- The clearest finished wins are `AIME`, `LiveCodeBench`, and `MMLU-Pro`.
- `GPQA` is a real ranking win over metadata, but fixed-threshold metrics are unstable because only `5%` of the test prompts are positive.
- `MATH-500` remains weak overall; it is a small ranking lift over metadata, not a clean threshold-quality win.

## Interpretation

- The earlier thread snippets were regression-only early checkpoints. The full locked run says the binary head is the more convincing part of this pair.
- Saying only "ensemble beats last layer" is too coarse:
  - for regression, that does not imply a metadata win;
  - for binary, the gain is real but still prevalence-sensitive on the rare-event datasets.
- The repo should now describe the regression result precisely:
  - `mean_relative_length` is still the stronger activation regression target relative to `last_layer`,
  - but this full-train pass does not show robust lift over the train-fit prompt-length control across the five-dataset suite.
- The repo should also describe the binary result precisely:
  - `majority_s_0.5` is still not a pure loop label,
  - but it is the cleaner finished activation-lift surface on this run.

## Bottom Line

The locked full-train item from `backlog.md` is now complete and reported in the same style as the earlier loop-label experiment notes.

What changed after actually running it:
- The binary head looks materially stronger than the regression head as a finished deployment-facing surface.
- The regression head still needs a more careful answer to the metadata question before it should be treated as a clear activation win.
- Future handoffs should cite this result note and the copied ledger directly instead of falling back to the early GPQA/AIME-only Slack checkpoints.
