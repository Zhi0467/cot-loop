# Qwen3 Loop-Trigger Attention Query Comparison

Last updated: 2026-04-14 21:43 UTC

## Object

This note now answers the loop-trigger question at **two** query positions on the saved March 22-23 Qwen3-1.7B instruct prompt-profile archives:

- `trigger_end`: the final token of the twentieth repeated `30`-gram
- `trigger_start`: the first token of that same twentieth repeated `30`-gram

Everything else is held fixed. The analyzed object is still:

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

There is no extra mid-prefix truncation inside either run. Each HF forward still sees the full prompt plus the completion prefix through the saved first loop trigger. The only thing that changes is which token inside the triggering copy is used as the query.

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

The dominant mismatch is still one hidden empty-text stop token on `finish_reason="stop"`. So both analyses remain exact prompt replay plus approximate completion replay. The important boundary is unchanged: the trigger prefix is recovered on `811 / 820` loop rows, which is strong enough for both query-position studies.

## How The Attention Analysis Is Defined

1. Recompute the loop trigger on the reconstructed completion token IDs with `find_ngram_loop_trigger(..., n=30, k=20)`.
2. Keep only rows where the recomputed `first_loop_prefix` exactly matches the saved `first_loop_prefix_length`.
3. Replay the model on:
   - `prompt_token_ids + completion_token_ids[:first_loop_prefix_length]`
4. Define two query positions on that same replay:
   - `trigger_end = prompt_len + trigger_end = total_prefix_length - 1`
   - `trigger_start = prompt_len + trigger_start`
5. For every layer and every head, capture the softmax attention from that one query token while preserving the model's masking behavior.
6. Partition key positions into disjoint bins:
   - `prompt`: all prompt positions
   - `current_trigger`: the current triggering `30`-gram, including self
   - `previous_loop`: all earlier copies of that same `30`-gram, excluding overlap with the current trigger span
   - `last_previous_loop`: the immediately preceding loop copy, again excluding overlap
   - `recent_nonloop`: the previous `256` completion positions that are in neither `previous_loop` nor `current_trigger`
   - `other_completion`: any remaining completion positions outside those bins
7. Summaries are defined as:
   - `mean_*_mass`: for one row, sum the attention weights on that bin, average across heads, then average those row-level values across the selected rows
   - `top1_fraction_*`: fraction of heads whose argmax lands in that bin

One important detail for the `trigger_start` read: the bins are defined over the **full** trigger span for consistency, but the model is still causal. So positions to the right of the query token get zero mass automatically. That means `current_trigger` at `trigger_start` is effectively dominated by the first trigger token/self, whereas `current_trigger` at `trigger_end` covers the full already-written triggering copy.

## Final-Layer Overall Comparison

The final-layer summary is layer `27` for every dataset.

| Query token | Prev-loop mass | Prompt mass | Current-trigger mass | Recent-nonloop mass | Top-1 prev-loop | Top-1 prompt | Top-1 current-trigger | Top-1 recent-nonloop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trigger_end` | `0.031` | `0.639` | `0.176` | `0.038` | `0.0003` | `0.874` | `0.114` | `0.0005` |
| `trigger_start` | `0.054` | `0.576` | `0.170` | `0.091` | `0.019` | `0.825` | `0.120` | `0.029` |

So moving the query from the end of the triggering copy to its first token does change the late-layer read:

- previous-loop mass rises from `0.031` to `0.054`
- previous-loop top-1 fraction rises from essentially zero (`0.0003`) to a small but real `0.019`
- prompt mass falls from `0.639` to `0.576`
- recent-nonloop mass rises from `0.038` to `0.091`
- current-trigger mass stays in the same rough band (`0.176` to `0.170`), but the interpretation changes because at `trigger_start` that bin is mostly self

The important part is that `trigger_start` is **less** prompt-dominant than `trigger_end`, but it is still prompt-dominant in the final layer.

## Trigger-End Final-Layer Read

| Dataset | Rows | Prev-loop mass | Prompt mass | Current-trigger mass | Recent-nonloop mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `121` | `0.025` | `0.671` | `0.173` | `0.039` |
| `AIME` | `30` | `0.038` | `0.627` | `0.176` | `0.027` |
| `MATH-500` | `68` | `0.029` | `0.634` | `0.185` | `0.039` |
| `MMLU-Pro` | `130` | `0.027` | `0.620` | `0.185` | `0.050` |
| `LiveCodeBench` | `462` | `0.034` | `0.637` | `0.174` | `0.034` |
| **Overall row-weighted** | **`811`** | **`0.031`** | **`0.639`** | **`0.176`** | **`0.038`** |

This is the original late-trigger read: overwhelmingly prompt-focused, with previous-loop top-1 attention essentially absent.

## Trigger-Start Final-Layer Read

| Dataset | Rows | Prev-loop mass | Prompt mass | Current-trigger mass | Recent-nonloop mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GPQA` | `121` | `0.035` | `0.608` | `0.171` | `0.095` |
| `AIME` | `30` | `0.073` | `0.564` | `0.163` | `0.073` |
| `MATH-500` | `68` | `0.061` | `0.584` | `0.156` | `0.092` |
| `MMLU-Pro` | `130` | `0.031` | `0.565` | `0.177` | `0.120` |
| `LiveCodeBench` | `462` | `0.063` | `0.570` | `0.170` | `0.083` |
| **Overall row-weighted** | **`811`** | **`0.054`** | **`0.576`** | **`0.170`** | **`0.091`** |

Final-layer head destinations move in the same direction:

- overall row-weighted top-1 fractions at `trigger_start`:
  - `prompt = 0.825`
  - `current_trigger = 0.120`
  - `previous_loop = 0.019`
  - `recent_nonloop = 0.029`
- by dataset, `top1_fraction_previous_loop` ranges from `0.0019` on `MMLU-Pro` to `0.0273` on `LiveCodeBench`

So the stricter first-token query does pick up more late-layer previous-loop attention, but prompt tokens are still the dominant target on every dataset.

## Mid-Stack Comparison

Earlier loop copies remain much more visible in the middle of the stack, and the peak layer stays `6` for every dataset under **both** query definitions.

| Query token | Peak layer | Prev-loop mass | Prompt mass | Current-trigger mass | Top-1 prev-loop | Top-1 prompt | Top-1 current-trigger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trigger_end` | `6` | `0.193` | `0.265` | `0.223` | `0.139` | `0.596` | `0.167` |
| `trigger_start` | `6` | `0.249` | `0.245` | `0.106` | `0.197` | `0.605` | `0.077` |

At `trigger_start`, the mid-stack previous-loop signal is plainly stronger than at `trigger_end`. The clearest dataset-level peak is `AIME`, where layer `6` reaches `prev-loop mass = 0.299`; the weakest is still `GPQA`, but even there layer `6` reaches `0.218`.

## Conclusion

The updated read is:

- the strong claim "Qwen3 is not paying attention to previous loops" is still too strong;
- the original narrower claim remains true at `trigger_end`: the final-layer state after the full triggering copy is written is overwhelmingly prompt-focused;
- the stricter `trigger_start` query is meaningfully different: previous-loop attention is stronger both in the final layer (`0.054` vs `0.031`) and in the middle of the stack (`0.249` vs `0.193`);
- but even at `trigger_start`, the final layer is still mostly prompt-focused rather than previous-loop-focused (`prompt mass = 0.576`, `top1_prompt = 0.825`).

So if the research question is literally "what is the model looking at on the **first token** of the final repeated copy?", the answer is:

- it is still looking mostly at the prompt in the final layer;
- previous loop copies are not absent, and they are more salient than they looked at `trigger_end`;
- the strongest previous-loop signal still lives in the middle of the stack rather than at the final layer.

The next still-open refinement is the one-token-earlier `trigger_start - 1` query plus matched non-loop controls.

## Future Rollouts

The saver-side fixes are already on this branch:

- prompt-profile archives now save exact `completion_token_ids` plus structured `loop_trigger`
- the rollout-stats collector now writes `__rollout_archive.jsonl.gz` with prompt/completion token IDs and trigger metadata

So this exact reconstruction gap should not recur on future rollouts.

## Artifact Bundle

Raw outputs now live in two sibling directories:

- `outputs/qwen3_loop_trigger_attention_full_20260414_rerun/` for `trigger_end`
- `outputs/qwen3_loop_trigger_attention_trigger_start_20260414/` for `trigger_start`

Main raw files in each bundle:

- `analysis_config.json`
- `reconstruction_summary.json`
- `selected_rows.jsonl`
- `attention_layer_means.csv`
- `attention_summary.json`
- `attention_per_sample.json`

The updated report PDF stays under the original report bundle:

- `outputs/qwen3_loop_trigger_attention_full_20260414_rerun/qwen3_loop_trigger_attention_full_20260414_rerun.pdf`
