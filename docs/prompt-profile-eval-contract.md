# Prompt-Profile Evaluation Contract

Last updated: 2026-03-25 00:23 UTC

## Predictor Object

- The prompt-profile predictor is not trying to predict an intrinsic "loop label" for a prompt in the abstract.
- It predicts prompt-level terminal statistics of repeated rollouts for one fixed model and one fixed decode policy, using prompt-prefill activations only.
- Every target below is defined per prompt from repeated rollouts under that fixed policy.

Notation:

- `L_r`: generated length of rollout `r`
- `E`: effective token budget for that prompt/run
- `n`: number of repeated rollouts for that prompt

## Prompt-Level Targets

- `majority_s_0.5 = 1[sum_r 1[L_r / E >= 0.5] > n / 2]`
- `p_loop = E_r[1{rollout r loops}]`
- `p_cap = E_r[1{rollout r hits max length}]`
- `mean_relative_length = E_r[L_r / E]`
- `loop_budget_share = E_r[1{rollout r loops} * (L_r / E)]`

These targets answer different questions:

- `majority_s_0.5` asks whether this prompt is frequently long.
- `p_loop` asks how often this prompt enters the loop regime we actually label.
- `p_cap` asks how often this prompt reaches the hard ceiling.
- `mean_relative_length` asks how much token budget this prompt tends to consume on average.

## What "Prompt-Length-Only Baseline" Means

This phrase had drifted across notes. The exact meaning is now explicit.

### Binary Targets

For binary heads such as `majority_s_0.5`, the prompt-length baseline is already a real one-feature scorer:

- input feature: raw `prompt_token_count` only
- no activations
- no MLP
- no hidden layer

Selection rule in the current exporter:

- choose orientation on the train split: `higher` or `lower` prompt length means "more positive"
- pick the orientation that maximizes train `PR-AUC`, tie-break by train `ROC-AUC`
- choose the threshold on that oriented 1D score that maximizes train `macro-F1`, tie-break by positive-class `F1`, then accuracy
- apply that fixed 1D rule to the held-out test prompts

The `effective_budget` baseline is defined the same way, but uses `effective_max_tokens` as the only feature.

### Continuous Targets

For the current five-dataset table on `p_loop`, `mean_relative_length`, and `loop_budget_share`, the "prompt-length baseline" is weaker and more limited:

- it is currently just the raw held-out association between prompt length and the target
- in practice that means `Spearman(prompt_token_count, target)` in the saved summaries
- it is not yet a trained regressor or calibrated metadata model

This is why the joint `prompt_length + effective_budget` baseline is still an open measurement item rather than a finished one. The next baseline pass needs true metadata-only models for:

- `prompt_length`
- `effective_budget`
- `prompt_length + effective_budget`

evaluated with the same held-out metrics as the activation probes.

## How To Read The Metrics

- `Spearman rank correlation`: correlation between the ranking induced by the predictor and the ranking induced by the realized prompt-level target across prompts. `1` is perfect monotone ranking, `0` is no monotone signal, negative means the ranking is reversed.
- `top-20% capture`: sort prompts by predicted score, keep the top 20% highest-scored prompts, and measure what fraction of the total target mass lies inside that bucket.
- For `p_loop`, target mass means the summed empirical loop probabilities across prompts.
- For `mean_relative_length`, target mass means the summed realized relative-length mass across prompts.
- `PR-AUC`, `ROC-AUC`, `macro-F1`, positive precision, and positive recall are binary-target metrics only.
- `Brier` and `MSE` are calibration/error guardrails. They matter, but they are not the main utility metric when the downstream use is prompt ranking or bad-prompt capture.

## What The Strong `majority_s_0.5` Baseline Means

A strong prompt-length baseline on `majority_s_0.5` does not mean the target is useless.

- If the operational goal is simply "catch prompts likely to produce degenerate rollouts," then a cheap metadata-only predictor can still be a valid solution.
- What it does mean is that `majority_s_0.5` is a weak target for claiming activation-specific lift beyond prompt geometry.

Why prompt length can do well on this head:

- the label is coarse: with `n = 4`, counts `0`, `1`, and `2` long rollouts all map to `0`, while counts `3` and `4` map to `1`
- the head is therefore a "frequently long" label, not a direct "frequently looped" label
- under a fixed budget and fixed decode policy, prompt difficulty and prompt length already move that long-rollout frequency substantially

Current evidence says this is not only a `GPQA` artifact:

- on the finished `AIME` seed-`0` split, prompt length alone already reaches `PR-AUC 0.8976` for `majority_s_0.5`

What is still unresolved:

- whether `0.5` is itself too easy, versus whether majority-threshold heads are generically too geometry-shaped
- whether `majority_s_0.5` top-risk prompts actually isolate degenerate behavior as well as `p_loop` or `p_cap`

That second question needs a direct bucket comparison on predicted high-risk prompts:

- empirical loop rate
- empirical max-length-hit rate
- empirical accuracy

## Current Decision Rule

- If the goal is "screen bad prompts by any cheap signal," a metadata baseline may already be acceptable.
- If the goal is "show that prompt-prefill activations add signal beyond prompt geometry," the probe must beat the strongest metadata-only baseline on the same held-out metric.

Current head recommendation:

- ship `mean_relative_length` as the main useful score
- keep `p_loop` alongside it as the cleaner failure-prox companion
- keep `majority_s_0.5` as a control and possible cheap operational screen, not as the main activation-lift claim
- only reopen direct `p_cap` if the next baseline-and-bucket checks still leave the binary-head choice unsettled
