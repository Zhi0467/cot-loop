# CoT Loop Detection

Research code for studying **chain-of-thought (CoT) loop** and **max-length** failures in reasoning language models, and for predicting those failures from internal model signals.

A sequence is considered *looped* when any `n`-gram appears `k` or more times in the generated token ids (default `n = 30`, `k = 20`). A *max-length hit* is a rollout that terminates by exhausting its decode budget rather than by emitting a stop token. Together, these two events cover most of what is meant by "the model got stuck instead of finishing."

This repository contains two complementary evidence streams:

1. **Probe line** — train classifiers and regressors on stacked prompt-prefill activations to predict per-prompt loop / long-rollout risk *before* any tokens are generated.
2. **Rollout-statistics line** — collect paired `thinking on` / `thinking off` rollouts on a fixed benchmark family under one shared sampling policy, then measure how often loops and cap hits actually occur.

The current canonical rollout-stats surface is a paired four-dataset `Qwen/Qwen3-1.7B` rebuild on `LiveCodeBench`, `TACO-hard`, `MATH level-5`, and `Omni-MATH >= 7`. Per-prompt archives preserve the full prompt, prompt token ids, per-rollout completion text, completion token ids, and raw row metadata, so the same rollouts can be relabeled for probe training, mechanism analysis, or steering without re-running the GPU work.

## Installation

Requires Python `>= 3.10`. Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

`vllm` and `flash_attn` are gated to non-macOS platforms; the probe-side CLIs (dataset building with Transformers-only prefill, training, evaluation, plotting) still work on macOS.

Training logs to Weights & Biases. Put your key in a local `.env`:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

## Quick start

### 1. Build a probe dataset (features + labels)

Extract per-prompt prefill activations, run rollouts with vLLM, and label each rollout with the loop detector:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset HuggingFaceH4/MATH-500 \
  --train-split test \
  --prompt-field problem \
  --model-preset openthinker3_1p5b \
  --out-dir outputs/probe_data/my_run
```

If `--test-dataset` is omitted, the test split defaults to the local `data/aime_2024_2025.jsonl` file (which expects `question` / `answer` fields). For a deterministic random split from a single dataset, pass identical train/test specs plus `--split-ratio`.

The default feature view is `last_token_all_layers_stack_final`: for each prompt, the last-prompt-token residual vector from every transformer block, stacked into `[num_layers, hidden]`. Other views (concat, mean, tail-64 strided, window deltas) are defined in `src/loop_probe/prefill.py`.

For prompt-level aggregate targets over repeated rollouts (`p_loop`, `p_cap`, `mean_relative_length`, or binary `majority_s_0.5`), build with:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> \
  --model-preset openthinker3_1p5b \
  --target-kind probability --profile-target p_loop --num-generations 10 \
  --out-dir outputs/probe_data/p_loop_run
```

### 2. Train a probe

```bash
python scripts/train_probe.py \
  --data-dir outputs/probe_data/my_run \
  --out-dir outputs/probe_runs/run1 \
  --probe-preset mlp \
  --wandb-project cot-loop-probe \
  --epochs 10 --batch-size 256
```

Default training slices the final layer out of the stacked `[layer, hidden]` view. Use `--classifier-mode ensemble` for a per-layer MLP head with vote or soft-probability aggregation. Probe presets are `linear` and `mlp`; MLP geometry can be overridden with `--mlp-hidden-dim`, `--mlp-depth`, `--mlp-dropout`.

Class imbalance is handled automatically via a `pos_weight` term in BCE-with-logits (binary) or via soft BCE / sigmoid-MSE (probability / regression targets).

### 3. Evaluate a saved checkpoint on another split or dataset

```bash
python scripts/eval_probe_checkpoint.py \
  --checkpoint outputs/probe_runs/run1/best.pt \
  --data-dir outputs/probe_data/other_eval \
  --split test
```

### 4. Run the end-to-end probe pipeline on SLURM

```bash
sbatch slurm/run_probe_train_e2e.sbatch
```

Override the defaults with environment variables: `MODEL_PRESET`, `TRAIN_DATASET`, `TRAIN_SPLIT`, `PROMPT_FIELD`, `PROBE_PRESET`, `MAX_NUM_SEQS`, `COMPLETION_BATCH_SIZE`, etc. See `slurm/README.md` for the full set.

### 5. Rollout-statistics suite

To (re)collect the paired `thinking on` / `thinking off` rollout-stats bundle on the canonical four-dataset `Qwen/Qwen3-1.7B` surface:

```bash
python scripts/launch_main_rollout_stats_suite.py \
  --output-root outputs/model_stats/main_rollout_stats_rebuild \
  --thinking-modes on,off --submit
```

The suite definition (model, sampling config, per-dataset contracts) lives in `src/loop_probe/main_rollout_stats_suite.py`. Per-prompt archives preserve prompt text, prompt token ids, rollout completion text, completion token ids, and raw row metadata so the same rollouts can drive later prompt-profile relabeling, probe training, and mechanism analysis.

Slurm launchers source `slurm/cache_env.sh`; Hugging Face model and dataset caches resolve to `/data/shared/huggingface` by default. Keep HF artifacts there, not under a user-specific `/data/scratch/${USER}/cache` or `/data/users/${USER}/cache` path.

For standalone single-model rollout generation and loop-metric summaries on one dataset:

```bash
python scripts/run_vllm_generate.py \
  --model-id Qwen/QwQ-32B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/qwq32b_metrics.csv \
  --tp 8

python scripts/compute_metrics.py \
  --generations path/to.jsonl \
  --out outputs/metrics.csv
```

## Model presets

Presets live in `src/loop_probe/configs.py` and can be overridden on the CLI (`--temperature`, `--max-tokens`, `--tp`, `--dp`, `--max-num-seqs`, …) or skipped entirely with `--model-id`.

| Preset | Model | TP | DP | Temperature | Max tokens |
|---|---|---|---|---|---|
| `qwq_32b` | `Qwen/QwQ-32B` | 8 | 1 | 0.0 | 30000 |
| `openthinker3_7b` | `open-thoughts/OpenThinker3-7B` | 1 | 8 | 0.0 | 30000 |
| `openthinker3_1p5b` | `open-thoughts/OpenThinker3-1.5B` | 1 | 8 | 0.0 | 30000 |

The rollout-stats suite pins its own contract (`Qwen/Qwen3-1.7B`, `temperature=0.2`, `num_generations=10`, `max_tokens=81920`, `max_model_len=40960`, `max_num_seqs=10`, `max_num_batched_tokens=4096`) and is not controlled through these presets.

## Loop and cap-hit definitions

- **Loop label**: any `n`-gram (default `n = 30`) appears `k` or more times (default `k = 20`) in the generated token ids of a single rollout.
- **Max-length hit**: the rollout reaches `max_model_len` on prompt-plus-generation.
- **Cap hit (prompt-profile)**: the generation reaches `effective_max_tokens` on the generation side only.
- **Prompt-level aggregates**: `p_loop`, `p_cap`, `mean_relative_length`, and the binary `majority_s_0.5` threshold on per-rollout relative length, each computed from repeated rollouts over the same prompt.

These are defined precisely in `docs/understand-where-loop-and-max-length-come-from.md`.

## Outputs

Probe-dataset build:

- `{out_dir}/train/shard-*.pt` — training shards (features + labels)
- `{out_dir}/test/shard-*.pt` — test shards
- `{out_dir}/manifest.json` — dataset metadata and configuration
- `{out_dir}/diagnostics/prompt_rollout_archive.jsonl` — per-prompt rollout archive (for repeated-rollout targets)

Probe training:

- `{out_dir}/best.pt` / `{out_dir}/last.pt` — best and final checkpoints
- `{out_dir}/metrics.jsonl` — per-epoch evaluation metrics
- `{out_dir}/best_metrics.json` — best-row summary for this run

Multi-seed SLURM summaries:

- `{out_run_dir}/seed_*/metrics.jsonl` — per-seed train/eval metrics
- `{out_run_dir}/seed_summary.{json,csv}` — aggregate mean / std across seeds
- `{out_run_dir}/probe_multiseed_curves.png` — aggregated train/eval curves

Rollout-stats suite:

- one collector JSON per `(dataset, thinking_mode)` run plus a `suite_manifest.json` under the chosen `--output-root`
- paired archive sidecars (`__prompt_profile.jsonl`, `__prompt_rollout_archive.jsonl`) for downstream relabeling

## Repository structure

```
cot-loop/
├── src/loop_probe/               # Core library (no CLI)
│   ├── configs.py                # Model presets
│   ├── hf_data.py                # Dataset loading
│   ├── prompt_builder.py         # Single source of truth for chat prompts
│   ├── prefill.py                # Prefill activation extraction + feature views
│   ├── rollout.py                # vLLM trajectory generation
│   ├── labeling.py               # Loop / cap-hit detection and prompt-level aggregation
│   ├── serialization.py          # Shard and manifest I/O
│   ├── dataloader.py             # PyTorch dataset / dataloader
│   ├── probes/                   # Probe architectures (linear, MLP, layerwise ensemble)
│   ├── train_utils.py            # Training utilities
│   ├── collector.py              # Repaired v2 rollout-statistics collector
│   ├── main_rollout_stats_suite.py  # Canonical paired thinking on/off four-dataset suite
│   └── adapters/                 # Per-benchmark rollout adapters (MATH, GPQA, MMLU-Pro, LiveCodeBench, TACO)
├── scripts/                      # CLI entry points
│   ├── build_probe_dataset.py
│   ├── train_probe.py
│   ├── eval_probe_checkpoint.py
│   ├── launch_main_rollout_stats_suite.py
│   ├── run_vllm_generate.py
│   ├── compute_metrics.py
│   └── ...                       # analysis, plotting, aggregation utilities
├── slurm/                        # SLURM batch scripts
├── data/                         # Local datasets (see data/README.md for the JSONL schema)
├── docs/                         # Design notes and reports
└── outputs/                      # Generated artifacts
```

## Notes

- Prefill feature extraction uses `transformers` and loads the full model on one GPU.
- Rollout generation uses `vLLM` for batched inference; `vllm` and `flash_attn` are disabled on macOS.
- Probe build and rollout generation share one chat-prompt constructor (`scripts/utils.build_prompt`) so probe features and rollouts see the exact same text.
- GPU memory requirements scale with model size and batch / sequence settings.
- This is research code; there is no automated test suite. Smoke-validate new changes with small `--train-max-samples` / `--test-max-samples` and short rollout limits before launching long jobs.

## Further reading

- `docs/understand-where-loop-and-max-length-come-from.md` — precise definitions of the loop, cap, and max-length events.
- `docs/main-four-dataset-rollout-rebuild-2026-04-23.md` — the current canonical rollout-stats rebuild contract.
- `docs/prompt-profile-eval-contract.md` — locked evaluation contract for the prompt-profile probe targets.
- `docs/prompt-profile-unified-report-2026-04-09.md` — unified prompt-profile report across regression, binary, and mechanism surfaces.
- `src/loop_probe/README.md` — module-level documentation for the probe library.
