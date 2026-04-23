# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project focus

Research repo for predicting chain-of-thought (CoT) loop / max-length failures from model internals. Two parallel evidence streams:

1. **Probe line** — train classifiers/regressors on prompt-prefill activations to predict loop risk.
2. **Rollout-statistics line** — measure how often looping / cap-hitting actually occurs across benchmark families under a common decode policy.

The active execution object on the rollout-statistics line is the paired thinking `on` / `off` four-dataset rebuild described in `docs/main-four-dataset-rollout-rebuild-2026-04-23.md`: `LiveCodeBench`, `TACO-hard`, `MATH level-5`, and `Omni-MATH >= 7`, each collected twice under a shared `Qwen/Qwen3-1.7B` contract with reusable prompt-rollout archives. The active sizing rule is the current suite contract: full `LiveCodeBench release_v6` (`1055`), `TACO-hard 1000 / 5536`, `MATH level-5 1000 / 2304`, and full `Omni-MATH >= 7` (`916`). Earlier March repairs, the `LiveCodeBench`-only reruns, and the mistaken `LiveCodeBench-extra` lane are historical debugging context — do not treat them as the current stage contract.

Before making non-trivial changes, read `roadmap.md`, `backlog.md`, `docs/main-four-dataset-rollout-rebuild-2026-04-23.md`, `docs/prompt-profile-eval-contract.md`, and `understand-where-loop-and-max-length-come-from.md`. These hold the current suite contract, the current split/target decisions (regression on `mean_relative_length`, binary `majority_s_0.5`), and the locked prompt-profile eval contract — do not redefine metrics, splits, or the rebuild dataset list ad hoc.

## Environment

- Python >= 3.10, managed with `uv` + `pyproject.toml` (never edit `uv.lock` by hand).
- Install: `uv sync`. Add a dep by editing `pyproject.toml` and rerunning `uv sync`.
- Training needs `WANDB_API_KEY` in `.env` (loaded by the train scripts).
- `vllm` and `flash_attn` are gated to non-macOS platforms; don't try to import them on darwin.

## Common commands

Build a probe dataset (features + labels):
```
python scripts/build_probe_dataset.py \
  --train-dataset <hf-id-or-jsonl> --train-split <split> --prompt-field <field> \
  --model-preset openthinker3_1p5b --out-dir outputs/probe_data/<name>
```
Omitting `--test-dataset` defaults the test set to `data/aime_2024_2025.jsonl` (hardcoded to `question`/`answer` fields). Use identical train/test specs with `--split-ratio` for a deterministic random split.

Prompt-profile targets (prompt-level aggregates over repeated rollouts):
- binary/probability: `--target-kind probability [--profile-target p_loop|p_cap] [--profile-tail-threshold 0.9] --num-generations 10`
- regression: `--target-kind regression --profile-target mean_relative_length --num-generations 10`

Train a probe:
```
python scripts/train_probe.py --data-dir <probe_data_dir> --out-dir outputs/probe_runs/<run> \
  --probe-preset mlp --wandb-project cot-loop-probe
```
Default slices the final layer out of the stacked `[layer, hidden]` view. Use `--classifier-mode ensemble` for layerwise voting; `--score-rule mean_prob` for soft aggregation. MLP overrides: `--mlp-hidden-dim`, `--mlp-depth`, `--mlp-dropout`.

Evaluate a saved checkpoint on another split/dataset:
```
python scripts/eval_probe_checkpoint.py --checkpoint <run>/best.pt --data-dir <probe_data_dir> --split test
```

End-to-end on SLURM (build + train, multi-seed):
```
sbatch slurm/run_probe_train_e2e.sbatch
```
Override knobs via env vars (`MODEL_PRESET`, `TRAIN_DATASET`, `TRAIN_SPLIT`, `PROMPT_FIELD`, `PROBE_PRESET`, `MAX_NUM_SEQS`, `COMPLETION_BATCH_SIZE`, …). See `slurm/README.md` for the full set.

Paired four-dataset rollout-stats rebuild (current canonical surface):
```
python scripts/launch_main_rollout_stats_suite.py \
  --output-root outputs/model_stats/main_rollout_stats_rebuild \
  --thinking-modes on,off --submit
```
The suite definition lives in `src/loop_probe/main_rollout_stats_suite.py` (shared `Qwen/Qwen3-1.7B` contract, `temperature=0.2`, `num_generations=10`, `max_tokens=81920`, `max_model_len=40960`, `max_num_seqs=10`, `max_num_batched_tokens=4096`). The launcher forwards `CONDA_ENV` / `CONDA_DEFAULT_ENV` into every submitted job and passes `THINKING_MODE` through explicitly — do not re-reintroduce ad-hoc thinking flags. `LiveCodeBench-extra` is intentionally removed; it is a strict subset of the `release_v6` `LiveCodeBench` run.

Rollout generation and loop-metric summary (standalone, not used by the current rebuild or the probe path):
```
python scripts/run_vllm_generate.py --model-id Qwen/QwQ-32B --data data/aime_2024_2025.jsonl --metrics-out outputs/qwq32b_metrics.csv --tp 8
python scripts/compute_metrics.py --generations path/to.jsonl --out outputs/metrics.csv
```

There is **no automated test suite**. Smoke-validate new code with small `--train-max-samples` / `--test-max-samples` and short rollout limits before launching long jobs.

## Architecture

### Pipeline (build_probe_dataset.py)

One linear flow, each stage owned by a module in `src/loop_probe/`:

1. **Load prompts** — `hf_data.py` reads an HF dataset or local JSONL, honors `--prompt-field`, and materializes `SampleRecord`s (`types.py`).
2. **Format chat prompts** — `prompt_builder.py` / `scripts/utils.build_prompt` is the single source of truth for model-formatted prompts; the builder **and** analysis scripts must route through it so probe features and rollouts see the same text.
3. **Prefill activation extraction** — `prefill.py` runs a Transformers forward pass with `output_hidden_states=True` and pools according to a **feature view**. "Activation" here means the per-layer post-block residual stream (`out.hidden_states[1:]`), not the MLP sublayer output. The canonical view is `last_token_all_layers_stack_final`: for each prompt, the last-prompt-token vector from every transformer block stacked into `[num_layers, hidden]`. Other views (concat, mean, tail-64 strided, window deltas) are defined in the same file.
4. **Rollout** — `rollout.py` drives vLLM to generate `num_generations` trajectories per prompt.
5. **Labeling / aggregation** — `labeling.py` applies the n-gram loop detector (`--loop-n 30 --loop-k 20` default: any 30-gram appearing ≥20× → `loop=1`). For repeated-rollout targets, rollouts are aggregated prompt-level into `p_loop`, `p_cap`, `s_t`, `mean_relative_length`, or `loop_budget_share`.
6. **Serialize** — `serialization.py` writes `train/shard-*.pt`, `test/shard-*.pt`, and `manifest.json` (plus `diagnostics/prompt_rollout_archive.jsonl` for repeated-rollout targets).
7. **Train** — `train_probe.py` uses `dataloader.py` + `probes/*` + `train_utils.py`. Loss is weighted BCE (binary), soft BCE (probability target), or sigmoid-MSE (regression target). Class imbalance is auto-handled via `pos_weight`.

### Feature views & probe modes — how they interact

The probe dataset stores a dict of named feature views; `train_probe.py` selects one view and one mode:

- `classifier_mode=last_layer` slices a single layer (`--classifier-layer`, default `-1`) out of a stacked `[L, H]` view. Use a flat view (`last_token` with a fixed `--feature-layer`) if you want a scalar index rather than slicing.
- `classifier_mode=ensemble` (in `probes/layerwise_ensemble_probe.py`) trains a per-layer head and combines them via `--vote-rule` / `--score-rule`. Only meaningful on stacked views.

The current headline probe surface is **ensemble, `h256 d1`, on the balanced-binary `majority_s_0.5` object**. Regression runs on the natural prompt-disjoint split with natural sampling. Don't mix those two regimes by accident — see `docs/prompt-profile-eval-contract.md`.

### Adapters

`src/loop_probe/adapters/` holds per-benchmark rollout adapters (MATH free-form, GPQA MC, MMLU-Pro MC, LiveCodeBench codegen). When adding a dataset family, add an adapter here rather than branching inside the collector — the rollout-statistics reports rely on consistent adapter output.

### Collector and rollout-stats suite

`collector.py` is the repaired v2 rollout-statistics path. Cross-dataset reports (`scripts/build_cross_dataset_rollout_report.py`, `scripts/build_model_rollout_report.py`) consume its output. If you touch the rollout schema, both the collector contract and those report builders need to stay in sync.

`src/loop_probe/main_rollout_stats_suite.py` is the canonical four-dataset suite scaffold that the current rebuild uses (paired thinking `on` / `off`, shared sampling config, reusable prompt-rollout archives). `scripts/launch_main_rollout_stats_suite.py` is the sbatch wrapper. The archive reuse contract is load-bearing downstream: every finished JSON must preserve `record_id`, full prompt text, `prompt_token_ids`, rollout `completion_text`, `completion_token_ids`, and raw `record_metadata` so the same rollouts can drive later prompt-profile relabeling, probe training, and steering without re-running the GPU work. Two runtime corrections are now baked into the suite and must stay that way: the TACO native grader returns top-level callables directly (it does not rebind them as `Solution` methods), and `BAAI/TACO` is loaded via the HF parquet surface `hf://datasets/BAAI/TACO/ALL/<split>-*.parquet`, not the retired `TACO.py` dataset script.

### Scripts vs library

- `src/loop_probe/` — reusable, no CLI.
- `scripts/` — thin CLI entry points; this is where argparse lives and where new experiments are added. Follow `scripts/utils.py`'s `build_prompt` whenever a new script generates or features text — don't reimplement prompt formatting.

### Model presets

Defined in `src/loop_probe/configs.py` (`MODEL_ROLLOUT_DEFAULTS`): `qwq_32b` (tp=8), `openthinker3_7b` (dp=8), `openthinker3_1p5b` (dp=8). Override any field via CLI (`--temperature`, `--max-tokens`, `--tp`, `--dp`, `--max-num-seqs`, …) or skip presets entirely with `--model-id`.

## Conventions

- 4-space indent, PEP 8, explicit CLI args + type hints matching existing scripts. No formatter/linter is configured — match surrounding style.
- Keep utilities and CLI entry points in separate files (utilities in `src/loop_probe/` or `scripts/utils.py`, entry points in `scripts/*.py`).
- Output naming is load-bearing: run folders under `outputs/probe_runs/`, metrics as `*_metrics.csv`, checkpoints `best.pt` / `last.pt`, per-epoch `metrics.jsonl`, figures under `outputs/`. Report-style artifacts live in dated folders (e.g. `outputs/prompt_profile_combined_audit_20260405/`).
- Git messages are short, lowercase, imperative (`added dp`, `fixed probe batching`). PR bodies should include reproduction command(s) and expected artifacts.
