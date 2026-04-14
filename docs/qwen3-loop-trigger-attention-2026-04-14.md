# Qwen3 Loop-Trigger Attention Full Rerun

Last updated: 2026-04-14 20:38 UTC

## Object

This note answers the current loop-trigger question on the saved March 22-23 Qwen3-1.7B instruct prompt-profile archives:

- when Qwen3 reaches the repeated `30`-gram that trips the `n=30`, `k=20` loop detector, what is the query state attending to?

This is the full rerun, not the earlier `14`-row bounded pilot. The analyzed object is:

- model: `Qwen/Qwen3-1.7B`
- loop definition: `n=30`, `k=20`
- selection rule: all rows whose reconstructed trigger prefix matches the saved `first_loop_prefix_length`, with `max_trigger_prefix=40960` and `max_samples_per_dataset=1000`
- selected rows: `811 / 820` loop rows
  - `GPQA 121`
  - `AIME 30`
  - `MATH-500 68`
  - `MMLU-Pro 130`
  - `LiveCodeBench 462`
- total prefix lengths on the selected rows: median `9846`, `p95 = 22321`, max `28830`

There is no extra mid-prefix truncation inside this rerun. Each HF forward sees the full prompt plus the completion prefix through the saved first loop trigger.

## Reconstruction Boundary

The March prompt-profile archives preserve exact `prompt_token_ids`, but they do **not** preserve exact `completion_token_ids`. For those older rows, the replay path still has to retokenize saved `completion_text`.

| Dataset | Rollouts | Loop rows | Exact retokenized length | Match if one hidden stop token is allowed | Trigger-prefix match |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `792` | `124` | `46 / 792` | `749 / 792` | `121 / 124` |
| `AIME` | `240` | `31` | `38 / 240` | `234 / 240` | `30 / 31` |
| `MATH-500` | `2000` | `68` | `38 / 2000` | `1991 / 2000` | `68 / 68` |
| `MMLU-Pro` | `3200` | `130` | `34 / 3200` | `3191 / 3200` | `130 / 130` |
| `LiveCodeBench` | `3200` | `467` | `356 / 3200` | `3168 / 3200` | `462 / 467` |
| **Overall** | **`9432`** | **`820`** | **`512 / 9432`** | **`9333 / 9432`** | **`811 / 820`** |

The dominant mismatch is still one hidden empty-text stop token on `finish_reason="stop"`. So this remains an exact prompt replay plus an approximate completion replay. The important boundary is that the trigger prefix is recovered on `811 / 820` loop rows, which is strong enough for a full trigger-slice attention study.

## How The Attention Analysis Is Defined

1. Recompute the loop trigger on the reconstructed completion token IDs with `find_ngram_loop_trigger(..., n=30, k=20)`.
2. Keep only rows where the recomputed `first_loop_prefix` exactly matches the saved `first_loop_prefix_length`.
3. Replay the model on:
   - `prompt_token_ids + completion_token_ids[:first_loop_prefix_length]`
4. Define the query token as:
   - `query_position = prompt_len + trigger_end = total_prefix_length - 1`
   - so this is the **final token of the twentieth repeated 30-gram**
   - this is **not** `trigger_start - 1` or `trigger_start`
5. For every layer and every head, capture the softmax attention from that one query token while preserving the model's masking behavior.
6. Partition key positions into disjoint bins:
   - `prompt`: all prompt positions
   - `current_trigger`: the current triggering `30`-gram, including self
   - `previous_loop`: all earlier copies of that same `30`-gram, excluding overlap with the current trigger span
   - `last_previous_loop`: the immediately preceding loop copy, again excluding overlap
   - `recent_nonloop`: the previous `256` completion positions that are in neither `previous_loop` nor `current_trigger`
   - `other_completion`: any remaining completion positions outside those bins
7. Summaries are then defined as:
   - `mean_*_mass`: for one row, sum the attention weights on that bin, average across heads, then average those row-level values across the selected rows
   - `top1_fraction_*`: fraction of heads whose argmax lands in that bin

So the measured object is:

- the next-token state **after the full triggering copy has already been written**

That makes this a defensible trigger-state read, but it is not yet the stricter "about to enter the loop" object.

## Final-Layer Read

The final-layer summary is layer `27` for every dataset.

| Dataset | Rows | Prev-loop mass | Prompt mass | Current-trigger mass | Recent-nonloop mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `121` | `0.025` | `0.671` | `0.173` | `0.039` |
| `AIME` | `30` | `0.038` | `0.627` | `0.176` | `0.027` |
| `MATH-500` | `68` | `0.029` | `0.634` | `0.185` | `0.039` |
| `MMLU-Pro` | `130` | `0.027` | `0.620` | `0.185` | `0.050` |
| `LiveCodeBench` | `462` | `0.034` | `0.637` | `0.174` | `0.034` |
| **Overall row-weighted** | **`811`** | **`0.031`** | **`0.639`** | **`0.176`** | **`0.038`** |

Final-layer head destinations tell the same story:

- overall row-weighted top-1 fractions:
  - `prompt = 0.874`
  - `current_trigger = 0.114`
  - `previous_loop = 0.0003`
  - `recent_nonloop = 0.0005`
- by dataset, `top1_fraction_previous_loop` is exactly `0` on `GPQA`, `AIME`, `MATH-500`, and `MMLU-Pro`, and only `0.00054` on `LiveCodeBench`

So on the exact object measured here, the final layer is overwhelmingly prompt-focused rather than previous-loop-focused.

## Mid-Stack Read

Earlier loop copies are not absent. They show up much more clearly in the middle of the stack. The dataset-level peak of `mean_prev_loop_mass` is layer `6` for every dataset:

| Dataset | Peak prev-loop layer | Prev-loop mass | Prompt mass | Current-trigger mass |
| --- | ---: | ---: | ---: | ---: |
| `GPQA` | `6` | `0.181` | `0.279` | `0.229` |
| `AIME` | `6` | `0.205` | `0.254` | `0.242` |
| `MATH-500` | `6` | `0.198` | `0.281` | `0.214` |
| `MMLU-Pro` | `6` | `0.179` | `0.262` | `0.235` |
| `LiveCodeBench` | `6` | `0.199` | `0.260` | `0.218` |
| **Overall row-weighted at layer 6** |  | **`0.193`** | **`0.265`** | **`0.223`** |

So the honest read is:

- earlier loop copies are present in the computation;
- they are strongest around layer `6` on this full rerun;
- but even there they are still not the dominant average target over prompt plus current-trigger mass.

## Conclusion

For the exact surface measured here:

- the strong claim "Qwen3 is not paying attention to previous loops" is too strong for the whole stack;
- the narrower claim "the final-layer trigger state is mostly prompt-focused, not previous-loop-focused" is supported by the full rerun;
- the current result does **not** yet answer the stricter "about to enter the loop" question, because the query token is the end of the triggering copy, not its start.

If we want the more literal pre-entry object next, the rerun should move the query to `trigger_start - 1` or `trigger_start`, ideally with matched non-loop controls.

## Future Rollouts

The saver-side fixes are already on this branch:

- prompt-profile archives now save exact `completion_token_ids` plus structured `loop_trigger`
- the rollout-stats collector now writes `__rollout_archive.jsonl.gz` with prompt/completion token IDs and trigger metadata

So this exact reconstruction gap should not recur on future rollouts.

## Artifact Bundle

The full rerun bundle is:

- `outputs/qwen3_loop_trigger_attention_full_20260414_rerun/`

Main files:

- `reconstruction_summary.json`
- `selected_rows.jsonl`
- `attention_layer_means.csv`
- `attention_summary.json`
- `attention_per_sample.json`
- `qwen3_loop_trigger_attention_full_20260414_rerun.tex`
- `qwen3_loop_trigger_attention_full_20260414_rerun.pdf`
