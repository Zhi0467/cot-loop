# Qwen3 Loop-Trigger Attention Pilot

Last updated: 2026-04-13 21:18 UTC

## Question

Wangzhi asked for a first implementable test of this hypothesis on the saved Qwen3-1.7B instruct prompt-profile archives:

- when the model is about to enter the loop regime that trips the `n=30`, `k=20` detector, is it paying attention to the previous loop copies?

This note answers that question only for a bounded pilot on the later March 22-23 prompt-profile archives. It does **not** claim exact rollout-time replay of the original completion token IDs, and it does **not** cover the long `8k+` trigger prefixes yet.

One correction matters up front: an earlier draft of this pilot had a per-layer wrapper capture bug in the attention script. I fixed that and reran the full bounded slice.

A later local review found one more bug that matters for the actual attention numbers: when repeated `30`-grams overlap, the old binning logic double-counted some token positions as both "previous loop" and "current trigger". I fixed that bug in the script too, but I have **not** rerun the remote archive sweep after that final fix in this checkout. So the numerical attention summaries from the earlier rerun are now withdrawn. Until the rerun lands, the durable parts of this note are:

- the reconstruction boundary;
- the definition of the bounded slice; and
- the rollout-saving changes for future runs.

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

The important current fact is not a number but a contract:

- the bounded pilot is still defined by the same `14` selected short-prefix rows listed above;
- the script now respects sliding-window masking, excludes overlapping current-trigger positions from the previous-loop bin, and consumes a `rollout_bundle.v1` bundle (`<base>.jsonl.gz`) directly via `probe.bundle_io.iter_bundle_rows`; the `__rollout_archive.jsonl.gz` / prompt-profile archive inputs it used to accept have been superseded by that bundle (migrate older bundles with `scripts/rollout/migrate_legacy_rollout_bundle.py`);
- the next rerun of this exact bounded slice is the right place to refresh the actual attention summary.

## Read

The honest current answer is therefore procedural rather than empirical:

- the saved archives are sufficient to define and replay the bounded trigger slice;
- the exact future-rollout saving path is now fixed, so this replay gap should not repeat on new runs;
- the clean answer to the actual attention question is pending the overlap-corrected rerun of this same slice.

## Code Changes For Future Rollouts

This task also patched both rollout-archive surfaces so future runs stop losing the exact objects this analysis needs.

### Prompt-profile archive path

`scripts/data/build_probe_dataset.py` now writes, for every rollout row in `diagnostics/prompt_rollout_archive.jsonl`:

- `completion_token_ids`
- `loop_trigger`
  - `ngram_start_positions`
  - `trigger_start`
  - `trigger_end`
  - `ngram_token_ids`

### Rollout-stats collector path

`scripts/rollout/collect_model_stats.py` now emits a `rollout_bundle.v1` pair
(`<base>.jsonl.gz` + `<base>.json`, see
[docs/reference/rollout-bundle-v1-schema.md](../../reference/rollout-bundle-v1-schema.md)). The bundle
replaces the older `__rollout_archive.jsonl.gz` sidecar and carries, per
prompt row:

- `prompt`
- `prompt_token_ids`
- `prompt_token_count`
- per-rollout under `rollouts[*]`:
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

- Report bundle: `outputs/weeks/2026-W16/qwen3_loop_trigger_attention_20260413/`
- Main raw summaries:
  - `reconstruction_summary.json`
  - `attention_summary.json`
  - `attention_layer_means.csv`
  - `attention_per_sample.json`
- Treat the current attention-number artifacts in this bundle as stale until the overlap-corrected rerun is refreshed.

## Next Honest Step

If this direction stays live, the next honest run is:

1. repeat the same trigger-attention measurement on longer trigger prefixes once GPU time is available;
2. add one matched counter-slice of non-loop or pre-trigger rows so "previous-loop attention" is compared against a clean control;
3. keep using saved exact `completion_token_ids` from the patched archive paths instead of reconstructed text.
