# Prompt-Profile Full Train Results

Last updated: 2026-04-04 02:58 UTC

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
- Regression eval pair: each held-out prompt `i` contributes one aligned pair `(\hat y_i, y_i)`, where `y_i` is that same prompt's saved `mean_relative_length` and `\hat y_i` is the probe prediction for that same prompt.
- Binary eval pair: each held-out prompt `i` contributes one aligned pair `(\hat s_i, y_i)`, where `y_i` is that same prompt's `majority_s_0.5` label and `\hat s_i` is the probe score for that same prompt.
- Important mechanism split:
  - for regression `mean_relative_length`, `ensemble` means one MLP per layer with `mean_prob` aggregation;
  - for binary `majority_s_0.5`, `ensemble` means one MLP per layer with `vote_fraction` aggregation over the per-layer 0/1 predictions.
- Metric roles in this note:
  - regression primary metric: `RMSE`;
  - regression `Spearman`: diagnostic only, measuring monotone association across the aligned held-out prompt pairs above;
  - binary primary ranking metric: `PR-AUC`, with fixed-threshold metrics secondary.
- "Metadata baseline" in this note means train-fit prompt-only scorers:
  - inputs: `prompt_token_count`, and separately `effective_max_tokens`;
  - excluded inputs: no activations, no prompt text, no dataset/source identity, no model identifier, no architecture features, no generated output;
  - regression baseline: standardized linear regression fit on the train split and evaluated on the held-out test split;
  - binary baseline: a 1D score rule fit on the train split, with direction chosen by train `PR-AUC` and threshold chosen by train macro-F1 / positive-F1 / accuracy, then evaluated once on held-out test.
- On this fixed run surface, `effective_budget` is constant at `30000`, so the only nontrivial metadata feature is prompt length. The joint regression control is therefore identical to the prompt-length-only control, and the binary `effective_budget` control is just a constant predictor.

Copied ledger for this run:
- `outputs/prompt_profile_full_train_locked_pair_20260404/remote_summary/`
- `outputs/prompt_profile_full_train_locked_pair_20260404/regression_summary.csv`
- `outputs/prompt_profile_full_train_locked_pair_20260404/binary_summary.csv`

## Main Results

### Regression `mean_relative_length`

The correct question here is prompt-level regression quality on held-out prompts. `RMSE` is the main fit metric. `Spearman` stays in the table only as a secondary diagnostic over the aligned prompt-level pairs above.

| Dataset | Ensemble `RMSE` | Last-layer `RMSE` | Prompt-length baseline `RMSE` | Ensemble `Spearman` (diag.) | Last-layer `Spearman` (diag.) | Prompt-length baseline `Spearman` (diag.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `0.160` | `0.164` | `0.165` | `0.419 +/- 0.045` | `0.175 +/- 0.225` | `0.116` |
| `AIME` | `0.183` | `0.189` | `0.176` | `0.639 +/- 0.090` | `0.483 +/- 0.138` | `0.676` |
| `MATH-500` | `0.162` | `0.169` | `0.154` | `0.351 +/- 0.199` | `0.158 +/- 0.426` | `0.461` |
| `MMLU-Pro` | `0.133` | `0.138` | `0.126` | `0.348 +/- 0.034` | `-0.019 +/- 0.293` | `0.393` |
| `LiveCodeBench` | `0.248` | `0.242` | `0.279` | `0.805 +/- 0.002` | `0.784 +/- 0.006` | `0.405` |

Clean read:
- On prompt-level error, ensemble beats `last_layer` on `GPQA`, `AIME`, `MATH-500`, and `MMLU-Pro`, but `last_layer` is slightly better on `LiveCodeBench`.
- Against the train-fit prompt-length baseline, the activation ensemble only wins on `GPQA` and `LiveCodeBench`.
- `AIME`, `MATH-500`, and `MMLU-Pro` remain prompt-length-dominated on this regression target.
- The old "ensemble wins on all five" phrasing was only true for the secondary `Spearman` diagnostic, not for the main prompt-level error metric.

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
- `MATH-500` remains weak overall; it is a small ranking lift over the prompt-length baseline, not a clean threshold-quality win.

## Interpretation

- The earlier thread snippets were regression-only early checkpoints. The full locked run says the binary head is the more convincing part of this pair.
- Saying only "ensemble beats last layer" is too coarse:
  - for regression, it depends on which metric you mean;
  - `Spearman` says the ensemble orders prompts better, but `RMSE` says `last_layer` still edges it on `LiveCodeBench`;
  - neither statement implies a prompt-length-baseline win;
  - for binary, the gain is real but still prevalence-sensitive on the rare-event datasets.
- The repo should now describe the regression result precisely:
  - `mean_relative_length` is still a usable activation regression target,
  - but this full-train pass does not show robust lift over the train-fit prompt-length control across the five-dataset suite.
- The repo should also describe the binary result precisely:
  - `majority_s_0.5` is still not a pure loop label,
  - but it is the cleaner finished activation-lift surface on this run.

## Bottom Line

The locked full-train item from `backlog.md` is now complete and reported in the same style as the earlier loop-label experiment notes.

What changed after actually running it:
- The binary head looks materially stronger than the regression head as a finished deployment-facing surface.
- The regression head should now be described with `RMSE` first and `Spearman` second; once that is done, the claim shrinks to "activation beats prompt length on `2 / 5` datasets."
- Future handoffs should cite this result note and the copied ledger directly instead of falling back to the early GPQA/AIME-only Slack checkpoints.
