# OLMo2 1B Fifty-Prompt Rerun - 2026-04-05

## Scope

This note is the larger follow-up to the bounded OLMo2 `1B` fallback ladder from `docs/weeks/2026-W14/olmo-degeneration-origin-audit-2026-04-04.md`.

Why this exists:
- Wangzhi objected that the earlier `8`-prompt slices were too small to carry stage conclusions.
- This rerun scales the same OLMo2 ladder to `50` prompts per dataset while keeping the same corrected prompt surfaces and the same bounded rollout-stat contract.
- The point is not to turn OLMo2 into a benchmark paper object. The point is to test whether the degeneration-origin story survives once the slice is large enough to stop looking like a pilot.

Short answer:
- the larger slice still says the big loop / max-length mass is already present in base;
- but it also says the later-stage story is not uniformly clean or monotone;
- `RLVR1` is usually the cleanest point in the ladder in the finished rerun;
- final instruct can re-accumulate loop / max-length mass on some datasets, especially `GPQA` and `MMLU-Pro`.

## Exact Object

Model ladder:
- `allenai/OLMo-2-0425-1B`
- `allenai/OLMo-2-0425-1B-SFT`
- `allenai/OLMo-2-0425-1B-RLVR1`
- `allenai/OLMo-2-0425-1B-Instruct`

Datasets:
- `MATH-500`
- `AIME`
- `GPQA`
- `MMLU-Pro`
- `LiveCodeBench release_v6`

Shared decode contract:
- `temperature=0.1`
- `top_p=0.95`
- `top_k=-1`
- `num_generations=10`
- `max_tokens=max_model_len=4096`
- `max_num_seqs=10`
- `max_num_batched_tokens=2048`
- `max_samples=50`

Reference-surface caveat:
- this OLMo2 ladder is the cheap same-collector fallback, not a literal replay of the older Qwen3 v2 horizon;
- the old Qwen reference object used `max_model_len=40960` and `max_tokens=81920`;
- OLMo2 `1B` only supports `max_model_len=4096`, so this rerun should be read as "does the stage pattern survive on a smaller same-family ladder?" rather than "did we exactly reproduce the old Qwen contract at lower cost?"

Prompt surfaces:
- base `MATH-500` / MCQ: native `raw`
- SFT / `RLVR1` / instruct `MATH-500` / MCQ: native `chat_template`
- `LiveCodeBench`: raw benchmark strings for every stage, with OLMo instruct checkpoints using the explicit `HFChatTemplate` LM style internally

Output root:
- `/data/scratch/murphy/outputs/cot-loop-detection/olmo2_1b_degeneration_origin_progression/bound50_temp0p1_gen10_ctx4096_topkneg1/`

Operational note:
- the first replay attempt died after generation because `scripts/rollout/collect_model_stats.py` wrote `top_p` / `top_k` from undefined names in `main()`;
- this report is only about the patched replay (`2257` to `2278`), not the failed write-path attempt.

Evaluation note:
- `Correct` below means rollout-level `num_correct` from the same collector family as the older Qwen3 rollout bundle, so each row is out of `500` generated rollouts rather than `50` prompts;
- for `LiveCodeBench`, the benchmark-native `pass@k` numbers are reported separately because per-rollout correctness is not the same object as prompt-level coding success.

## What The Fifty-Prompt Rerun Changed

Relative to the earlier `8`-prompt OLMo2 fallback:
- the main base-versus-later split survives;
- the simple "later stages are almost clean" read does not;
- the completed `AIME`, `GPQA`, and `MMLU-Pro` rows show meaningful residual degeneration after SFT and even after final instruct;
- the cleanest later-stage point is usually `RLVR1`, not automatically the final instruct checkpoint.

So the `8`-prompt note should now be treated as a pilot that pointed in the right direction on the coarse base-versus-later question, but understated how uneven the later-stage behavior still is.

## Fifty-Prompt Ladder

All rows below are `50 prompts x 10 generations = 500` rollouts.

| Stage | Dataset | Correct | Looped | Max-length hits | Avg generation length |
| --- | --- | --- | --- | --- | --- |
| base | `MATH-500` | `13 / 500` | `189 / 500` | `211 / 500` | `1749.7` |
| base | `AIME` | `0 / 500` | `224 / 500` | `240 / 500` | `1948.7` |
| base | `GPQA` | `23 / 500` | `256 / 500` | `308 / 500` | `2418.5` |
| base | `MMLU-Pro` | `12 / 500` | `235 / 500` | `313 / 500` | `2482.0` |
| base | `LiveCodeBench` | `0 / 500` | `117 / 500` | `475 / 500` | `3064.0` |
| SFT | `MATH-500` | `71 / 500` | `13 / 500` | `34 / 500` | `808.8` |
| SFT | `AIME` | `0 / 500` | `49 / 500` | `79 / 500` | `1532.9` |
| SFT | `GPQA` | `94 / 500` | `36 / 500` | `45 / 500` | `622.4` |
| SFT | `MMLU-Pro` | `44 / 500` | `15 / 500` | `15 / 500` | `305.6` |
| SFT | `LiveCodeBench` | `0 / 500` | `37 / 500` | `48 / 500` | `456.7` |
| `RLVR1` | `MATH-500` | `99 / 500` | `9 / 500` | `7 / 500` | `560.4` |
| `RLVR1` | `AIME` | `7 / 500` | `8 / 500` | `15 / 500` | `1044.0` |
| `RLVR1` | `GPQA` | `41 / 500` | `5 / 500` | `11 / 500` | `393.2` |
| `RLVR1` | `MMLU-Pro` | `10 / 500` | `22 / 500` | `24 / 500` | `409.0` |
| `RLVR1` | `LiveCodeBench` | `1 / 500` | `4 / 500` | `4 / 500` | `437.1` |
| instruct | `MATH-500` | `99 / 500` | `12 / 500` | `15 / 500` | `610.6` |
| instruct | `AIME` | `0 / 500` | `27 / 500` | `36 / 500` | `1109.0` |
| instruct | `GPQA` | `53 / 500` | `21 / 500` | `44 / 500` | `677.7` |
| instruct | `MMLU-Pro` | `18 / 500` | `70 / 500` | `93 / 500` | `991.7` |
| instruct | `LiveCodeBench` | `0 / 500` | `3 / 500` | `2 / 500` | `421.3` |

## LiveCodeBench Native Metrics

These are the benchmark-native coding metrics, not the rollout-stat `num_correct` field.

| Stage | pass@1 | pass@10 | Notes |
| --- | --- | --- | --- |
| base | `0.0` | `0.0` | degeneration is severe here too: `117 / 500` loops and `475 / 500` max hits |
| SFT | `0.0` | `0.0` | native OLMo path is working, but coding success is zero on this slice |
| `RLVR1` | `0.002` | `0.02` | native OLMo path is working, but coding success is still low |
| instruct | `0.0` | `0.0` | native OLMo path is working, but coding success is zero on this slice |

## Current Read

What is already stable:
- base is still the dominant source of degeneration mass on this object;
- base is heavily degenerate on all four completed non-code datasets, not just on one cherry-picked slice;
- SFT does not introduce the behavior from a clean base; it reduces a much worse base regime, but still leaves substantial residual mass on some datasets;
- `RLVR1` is the cleanest later-stage checkpoint on most datasets in the finished rerun;
- final instruct is mixed rather than a monotone improvement:
  - it is much cleaner than base;
  - it is often cleaner than SFT;
  - but it is not uniformly cleaner than `RLVR1`, and on `MMLU-Pro` it is much worse than both SFT and `RLVR1`.
- `LiveCodeBench` makes the base-origin split especially stark: base hits max length on `475 / 500` rollouts there, while SFT drops to `48 / 500`, `RLVR1` to `4 / 500`, and instruct to `2 / 500`.
- The native coding metric stays weak across the whole `1B` ladder (`pass@10 = 0.0` for base, SFT, and instruct; `0.02` for `RLVR1`), so `LiveCodeBench` is useful here mainly as a degeneration surface rather than a capability-ranking surface.

## Visualization Follow-up

The finished `50`-prompt ladder now has a direct figure bundle too:

- progression PDF:
  - `outputs/weeks/2026-W15/olmo2_1b_progression_bound50_20260406/olmo2_1b_progression_bound50.pdf`
- line charts:
  - `outputs/weeks/2026-W15/olmo2_1b_progression_bound50_20260406/figures/progression_rates.png`
  - `outputs/weeks/2026-W15/olmo2_1b_progression_bound50_20260406/figures/progression_lengths.png`
- within-stage overlap views:
  - `outputs/weeks/2026-W15/olmo2_1b_progression_bound50_20260406/figures/stage_overlap_sankey.png`
  - `outputs/weeks/2026-W15/olmo2_1b_progression_bound50_20260406/figures/stage_overlap_composition.png`

Those figures are derived from the saved `50`-prompt stage JSONs, not reconstructed manually from the table above.

What this changes relative to the earlier `8`-prompt pilot:
- the pilot was directionally right that base is the main degeneration source;
- the pilot was too optimistic about how clean the later-stage rows were, especially once `AIME`, `GPQA`, and `MMLU-Pro` are given `50` prompts instead of `8`.

What this rerun still does not prove:
- This is still the cheap OLMo2 fallback ladder, not a same-family OLMo3 `base -> SFT -> RLVR` result.
- `LiveCodeBench` capability is weak across the whole `1B` ladder, so that row should be read mainly for loop / max-length behavior rather than as a benchmark-capability claim.

## Artifact Locations

- Durable working note: `docs/weeks/2026-W14/understand-where-loop-and-max-length-come-from.md`
- Earlier smaller audit: `docs/weeks/2026-W14/olmo-degeneration-origin-audit-2026-04-04.md`
- PDF companion: `outputs/weeks/2026-W14/olmo2_1b_fifty_prompt_rerun_20260405/olmo2_1b_fifty_prompt_rerun_20260405.pdf`
- PDF source: `outputs/weeks/2026-W14/olmo2_1b_fifty_prompt_rerun_20260405/olmo2_1b_fifty_prompt_rerun_20260405.tex`
