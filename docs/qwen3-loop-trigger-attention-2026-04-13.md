# Qwen3 Loop-Trigger Attention Pilot

Last updated: 2026-04-13 21:18 UTC

## Question

Wangzhi asked for a first implementable test of this hypothesis on the saved Qwen3-1.7B instruct prompt-profile archives:

- when the model is about to enter the loop regime that trips the `n=30`, `k=20` detector, is it paying attention to the previous loop copies?

This note answers that question only for a bounded pilot on the later March 22-23 prompt-profile archives. It does **not** claim exact rollout-time replay of the original completion token IDs, and it does **not** cover the long `8k+` trigger prefixes yet.

One correction matters up front: an earlier draft of this pilot had a per-layer wrapper capture bug in the attention script. I fixed that and reran the full bounded slice; all numbers below are from the corrected rerun.

## Commit Audit

- I re-fetched `upstream/main` on 2026-04-13 21:18 UTC after Wangzhi's follow-up.
- The current upstream-main head is `d10155c` (`added attention scouting script`).
- So `main` now **does** carry the basic attention-analysis helper.
- The replay/saver fixes and the corrected attention-summary logic in this note still live on the task branch / PR `#10`; they were not part of `d10155c`.

## Reconstruction Boundary

The saved March 22-23 prompt-profile archives are still the right object for this pilot because they preserve:

- `prompt_token_ids`
- `completion_text`
- `finish_reason`
- `length`
- `first_loop_prefix_length`

What they still do **not** preserve is the exact `completion_token_ids`.

The reconstruction pass over the full saved archive family found:

| Dataset | Rollouts | Loop rows | Exact retokenized length | Match if one hidden stop token is allowed | Trigger-prefix match |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `792` | `124` | `46 / 792` | `749 / 792` | `121 / 124` |
| `AIME` | `240` | `31` | `38 / 240` | `234 / 240` | `30 / 31` |
| `MATH-500` | `2000` | `68` | `38 / 2000` | `1991 / 2000` | `68 / 68` |
| `MMLU-Pro` | `3200` | `130` | `34 / 3200` | `3191 / 3200` | `130 / 130` |
| `LiveCodeBench` | `3200` | `467` | `356 / 3200` | `3168 / 3200` | `462 / 467` |

The main pattern is simple:

- for `finish_reason="stop"`, the saved length is usually `retokenized_length + 1`;
- Qwen3's configured EOS tokens are `151645 = <|im_end|>` and `151643 = <|endoftext|>`;
- both decode to empty text, so they explain the one-token drift without being recoverable from the saved text itself.

So the exact completion-ID claim is still false, but the trigger-prefix claim is strong:

- retokenizing the saved text recovers the same loop trigger on essentially every loop row.

That is enough for a bounded attention pilot on the trigger prefix.

## Attention Pilot

I selected the shortest exact-trigger rows with total prefix length at most `4096`:

- `GPQA`: `3` rows (`712`, `1598`, `1684`)
- `AIME`: `2` rows (`3686`, `3870`)
- `MATH-500`: `3` rows (`409`, `805`, `1348`)
- `MMLU-Pro`: `3` rows (`2034`, `2048`, `2112`)
- `LiveCodeBench`: `3` rows (`1544`, `1662`, `2554`)

The attention probe looks only at the final token of the triggering `30`-gram copy. For that trigger token, it measures:

- attention mass on **all earlier copies** of the same loop `30`-gram;
- attention mass on the **prompt**;
- attention mass on the **current trigger span** itself;
- where each head's top-1 attention destination lands.

The cleanest read is the **final layer**:

| Dataset | Rows | Final-layer mass on previous loop spans | Prompt mass | Current-trigger mass |
| --- | ---: | ---: | ---: | ---: |
| `GPQA` | `3` | `0.038` | `0.755` | `0.154` |
| `AIME` | `2` | `0.035` | `0.724` | `0.170` |
| `MATH-500` | `3` | `0.102` | `0.654` | `0.192` |
| `MMLU-Pro` | `3` | `0.048` | `0.657` | `0.203` |
| `LiveCodeBench` | `3` | `0.112` | `0.625` | `0.224` |
| **Overall** | `14` | **`0.069`** | **`0.680`** | **`0.190`** |

Head destinations make the same point even more strongly:

- across all `14` rows, `85.3%` of final-layer heads put their top-1 attention destination on the **prompt**
- `14.7%` put top-1 on the **current trigger / self**
- `0%` put top-1 on **earlier loop copies**

So the corrected late-layer read is prompt-dominant, not previous-loop-dominant.

One more caution matters here. A later local review found that the attention wrapper also needed to respect sliding-window masking for checkpoints that use hybrid attention. I patched the script for that case, but I have **not** rerun the remote archive sweep after that final code fix in this checkout. So this note intentionally keeps only the bounded **final-layer** headline above as current. I am not making any intermediate-layer or "best layer" claim here until that rerun is refreshed.

## Read

What the pilot supports:

- if the claim is specifically about the **final layer** on this short exact-trigger slice, the model is mostly looking back to the **prompt**, not to earlier loop copies
- previous-loop mass in the final layer is still nonzero, especially on `LiveCodeBench` and `MATH-500`, but it is much smaller than prompt mass on every dataset in this bounded pilot

What the pilot does **not** yet support:

- it does not support any whole-stack or intermediate-layer claim until the sliding-window-corrected rerun is refreshed;
- it does not prove that prompt-focused final-layer attention is the causal reason the loop continues
- it does not prove that previous-loop attention is the causal reason the loop continues;
- it does not show the same pattern on the long `8k+` trigger rows;
- it does not tell us whether the important signal is "prompt vs previous loop" or "prompt vs local current-trigger span" without a matched control slice.

So the honest current answer is:

- **Yes**, on this short exact-trigger slice the **final layer** is overwhelmingly prompt-focused rather than previous-loop-focused.
- **Unknown for the whole stack**, because I am no longer treating the older intermediate-layer rows as current after the final sliding-window fix.
- So the current bounded read is narrower than the previous draft: the late-layer trigger token is not dominated by previous-loop attention on this slice.

## Example Rows

- `GPQA`, sample `36`, rollout `3`, total prefix `712`:
  - final layer: previous-loop mass `0.030`, prompt mass `0.848`, current-trigger mass `0.099`
- `MATH-500`, sample `437`, rollout `2`, total prefix `409`:
  - final layer: previous-loop mass `0.104`, prompt mass `0.664`, current-trigger mass `0.167`
- `LiveCodeBench`, sample `132`, rollout `0`, total prefix `1544`:
  - final layer: previous-loop mass `0.077`, prompt mass `0.662`, current-trigger mass `0.224`

## Code Changes For Future Rollouts

This task also patched both rollout-archive surfaces so future runs stop losing the exact objects this analysis needs.

### Prompt-profile archive path

`scripts/build_probe_dataset.py` now writes, for every rollout row in `diagnostics/prompt_rollout_archive.jsonl`:

- `completion_token_ids`
- `loop_trigger`
  - `ngram_start_positions`
  - `trigger_start`
  - `trigger_end`
  - `ngram_token_ids`

### Rollout-stats collector path

`scripts/collect_model_stats.py` now writes a compressed sidecar
`__rollout_archive.jsonl.gz` beside the aggregate JSON output. Each prompt row contains:

- prompt text
- `prompt_token_ids`
- `prompt_token_count`
- per-rollout:
  - `completion_text`
  - `completion_token_ids`
  - `completion_token_count`
  - `finish_reason`
  - `first_loop_prefix_length`
  - `loop_trigger`

Two review-caught implementation bugs were fixed before publish:

- resume-from-`LiveCodeBench` checkpoint now preserves the rollout sidecar that matches the resumed checkpoint instead of dropping it
- the collector now spills prompt-rollout archive chunks batch-by-batch, so it does not need end-of-run RAM proportional to the whole rollout archive

That closes the exact replay gap for future rollout-stat runs instead of forcing retrospective reconstruction from text.

## Artifacts

- Report bundle: `outputs/qwen3_loop_trigger_attention_20260413/`
- Main raw summaries:
  - `reconstruction_summary.json`
  - `attention_summary.json`
  - `attention_layer_means.csv`
  - `attention_per_sample.json`
- Treat non-final-layer rows in `attention_layer_means.csv` / `attention_per_sample.json` as provisional until the sliding-window-corrected rerun is refreshed.

## Next Honest Step

If this direction stays live, the next honest run is:

1. repeat the same trigger-attention measurement on longer trigger prefixes once GPU time is available;
2. add one matched counter-slice of non-loop or pre-trigger rows so "previous-loop attention" is compared against a clean control;
3. keep using saved exact `completion_token_ids` from the patched archive paths instead of reconstructed text.
