# MCQ Parser Pilot Summary

Date: 2026-03-16 01:31 UTC

## Bottom line

Keep the JSON-answer prompt and the tightened terminal parser.

- On the old freeform raw subsets, the tightened parser still recovers clear terminal answers that the brittle `Answer: X` rule missed, but it no longer credits provisional length-truncated reasoning such as `the answer is X` when the model kept talking after that line.
- On the new JSON-contract pilots, strict terminal JSON and the tightened terminal parser are identical on `MMLU-Pro` and differ by only one boxed stop completion on `GPQA`. So the new parser is not inflating the new-contract numbers.
- `GPQA` is still genuinely weak under the current rollout runtime even after the parser fix. The dominant failure mode is now `length` termination, not parser loss.

## Old freeform contract re-score

These rows reuse the earlier raw audit outputs and rescore them with the tightened terminal parser.

| Dataset | Rows | Old strict `Answer: X` | Tightened terminal parser | Tightened `length`-correct |
| --- | ---: | ---: | ---: | ---: |
| GPQA | 160 | 11 / 160 (6.88%) | 20 / 160 (12.50%) | 0 |
| MMLU-Pro | 80 | 9 / 80 (11.25%) | 16 / 80 (20.00%) | 0 |

Interpretation:

- The old parser materially undercounted obvious stopped answers.
- The tightened parser fixes that stopped-answer loss.
- The tightened parser does **not** rescue any length-truncated rows on these old subsets, by design.

## New JSON-contract pilots

These are fresh bounded pilots under the current JSON-answer prompt, `temperature=0.2`, `num_generations=10`, `max_tokens=4096`.

| Dataset | Rows | Old strict `Answer: X` | Strict JSON | Tightened terminal parser | Stop / Length |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPQA | 160 | 0 / 160 (0.00%) | 32 / 160 (20.00%) | 33 / 160 (20.63%) | 59 / 101 |
| MMLU-Pro | 80 | 0 / 80 (0.00%) | 25 / 80 (31.25%) | 25 / 80 (31.25%) | 60 / 20 |

Prompt-level reference points:

- GPQA `sample0` accuracy: 3 / 16 (18.75%).
- MMLU-Pro `sample0` accuracy: 2 / 8 (25.00%).

Parser-alignment details:

- `MMLU-Pro`: every parsed answer is terminal JSON. Strict JSON and tightened terminal parsing are exactly identical. There are zero fallback-only recoveries.
- `GPQA`: strict JSON and tightened terminal parsing differ by only one stopped boxed answer (`\\boxed{B}` after a `Conclusion` block). Every other parsed answer is terminal JSON.

## Interpretation

1. The parser decision is now clear.

   The earlier permissive tail scan was too loose for the old freeform contract because it could recover provisional `answer is X` lines from unfinished traces. The tightened parser fixes that. Under the new JSON contract, the parser is almost irrelevant because the model usually emits terminal JSON when it actually finishes.

2. The old `GPQA` and `MMLU-Pro` report rows were bad for two different reasons.

   The old freeform contract had a real parser bug. But that parser bug was not the whole story, especially for `GPQA`.

3. `MMLU-Pro` now looks like a parser-contract success.

   The bounded pilot moved from a broken freeform accounting surface to a clean JSON surface with 31.25% rollout accuracy on the same pilot scale.

4. `GPQA` still looks unhealthy after the parser repair.

   Even under the repaired JSON contract, `GPQA` is only 20.00%-20.63% on the bounded rollout slice, with 101 / 160 rollouts ending by `length`. So the remaining issue is no longer parser loss. It is the current runtime contract: decode budget, thinking trace length, or a broader benchmark-mismatch issue.

## Recommendation

- Commit to the JSON-answer prompt and the tightened terminal parser.
- Do **not** interpret the current `GPQA` rollout stat as a trustworthy benchmark-style MC accuracy number.
- If benchmark-comparable `GPQA` matters, rerun it under a benchmark-native evaluation setup with a contract that avoids the current 4096-token truncation pattern. The next thing to vary is the generation/runtime policy, not the parser.
