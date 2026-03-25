# CoT Loop Detection via Probe Classifiers

This repository studies whether chain-of-thought loop risk is predictable from internal activations and rollout telemetry. The original workflow centered on prompt-prefill last-token probes, and the current active phase extends that line with repaired cross-dataset rollout-statistics audits so the prefill/completion findings can be checked against real loop behavior across multiple benchmark families.

## Overview

The project now has two active evidence streams:
- the probe line asks whether loop risk is detectable from stacked prompt-prefill activations, either by slicing one layer or by voting across all layers;
- the rollout-statistics line measures how often looping and max-length hits actually occur under the repaired v2 collector contract across `MATH-500`, `AIME`, `GPQA`, `MMLU-Pro`, and `LiveCodeBench`.

Latest status:
- the best prefill-only arm is still the Round 6 all-layer last-token anchor; later metadata-aware prefill rounds did not overturn the completion-view advantage.
- the common-policy rollout-statistics bundle is now refreshed under one shared decode policy (`temperature=0.2`, `num_generations=10`, and `max prompts <= 800` where applicable) across `MATH-500`, `AIME`, `GPQA`, capped `MMLU-Pro`, and capped `LiveCodeBench release_v6`.
- the repaired MC rows are materially different from the stale pre-refresh bundle: `GPQA` now reports `34.5%` rollout success instead of `3.1%`, and `MMLU-Pro` now reports `65.2%` instead of `14.2%`, both under the terminal JSON-answer contract.
- the readable common-policy artifact is `outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/qwen3_1p7b_cross_dataset_rollout_report.pdf`, while the separate benchmark-style `GPQA` calibration lives in `outputs/gpqa_json_official_temp0p6_gen1/` and should not be conflated with the shared-policy table.
- one explicit caveat remains on that recovered `LiveCodeBench` block: the original job crashed after grading and before writing the final JSON, and replay-based repair did not reproduce the stored generations exactly enough to recover `avg_first_loop_prefix_length`. That single metric therefore remains `null` for the recovered capped run.
- the current prompt-profile question is now "which prompt-level terminal statistic is most useful to predict from prompt-prefill activations under a fixed model and decode policy?" rather than "which binary loop threshold should we force as the main label?" The current default bundle is `mean_relative_length` plus `p_loop`.
- `majority_s_0.5` is still kept as a control and possible cheap degenerate-rollout screen, but it is too prompt-length-shaped on `AIME` to be the main activation-lift claim. The exact target, baseline, and metric definitions now live in `docs/prompt-profile-eval-contract.md`.
- the current docs now lock one distinction that had kept drifting in-thread:
  - for binary `majority_s_0.5`, prompt length already means a true one-feature held-out scorer;
  - for the five-dataset continuous-head table, prompt length is still only raw held-out association, not yet a trained metadata-only predictor, so those rows should not be described as a metadata model "working."
- the current open measurement work is therefore concrete rather than vague:
  - add trained metadata-only baselines for `prompt_length`, `effective_budget`, and `prompt_length + effective_budget`;
  - compare the top predicted-risk prompts from `majority_s_0.5`, `p_loop`, and `mean_relative_length` on actual loop rate, cap-hit rate, and accuracy.

**Workflow:**
1. Build model-formatted chat prompts (shared `utils.build_prompt` source)
2. Extract stacked last-token activations from prompt-prefill states
3. Generate rollout trajectories and label them (looped vs not-looped)
4. Train a binary probe classifier on the precomputed features
5. Evaluate the probe's ability to predict looping behavior
6. For cross-dataset validation, run the rollout-statistics collector and rebuild the cross-dataset report from the authoritative stats bundle; the current repaired v2 bundle includes refreshed `GPQA` / `MMLU-Pro` JSON-answer rows plus the recovered capped `LiveCodeBench` block with the prefix-length caveat noted above

## Quick Start

### Installation

Requires Python >= 3.10.

```bash
uv sync
```

### Environment Setup

Create a `.env` file with your W&B API key:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

### Build Probe Dataset

Extract features and labels from train/test datasets:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset my_org/my_dataset \
  --train-split train \
  --test-dataset my_org/my_dataset \
  --test-split test \
  --prompt-field prompt \
  --model-preset qwq_32b \
  --out-dir outputs/probe_data
```

If `--test-dataset` is omitted, it defaults to local `data/aime_2024_2025.jsonl`:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset my_org/my_dataset \
  --train-split train \
  --prompt-field prompt \
  --model-preset openthinker3_7b \
  --out-dir outputs/probe_data
```

For this default local test file, loader behavior is hardcoded to use
`question` as prompt text and require `answer` on every row.

Optional: if you want a random split of one dataset, pass identical train/test specs
and use `--split-ratio`.

The default dataset stores one stacked feature view:
`last_token_all_layers_stack_final`, with per-sample shape `[layer, hidden]`.

```bash
python scripts/build_probe_dataset.py \
  --train-dataset HuggingFaceH4/MATH-500 \
  --train-split test \
  --test-dataset data/aime_2024_2025.jsonl \
  --test-split test \
  --prompt-field problem \
  --model-preset openthinker3_1p5b \
  --out-dir outputs/probe_data/openthinker3_1p5b_layers_stack
```

For balanced train/test probes after label construction:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset SuperSecureHuman/competition_math_hf_dataset \
  --train-split train \
  --train-max-samples 1800 \
  --test-dataset SuperSecureHuman/competition_math_hf_dataset \
  --test-split test \
  --test-max-samples 600 \
  --prompt-field problem \
  --model-preset openthinker3_1p5b \
  --balance-train downsample \
  --balance-test downsample \
  --out-dir outputs/probe_data/openthinker3_balanced_layers_stack
```

### Train Probe

```bash
python scripts/train_probe.py \
  --data-dir outputs/probe_data \
  --out-dir outputs/probe_runs/run1 \
  --wandb-project cot-loop-probe \
  --epochs 10 \
  --batch-size 256
```

By default, training slices the final layer from the stacked dataset view.

Available probe presets:
- `linear`
- `mlp` (default; configurable width/depth, defaults are in `src/loop_probe/configs.py`)

Optional MLP overrides:
- `--mlp-hidden-dim <int>`
- `--mlp-depth <int>`
- `--mlp-dropout <float>`

### Evaluate Saved Checkpoints on Another Split/Dataset

Torch probe checkpoint (`best.pt` / `last.pt`):

```bash
python scripts/eval_probe_checkpoint.py \
  --checkpoint outputs/probe_runs/run1/best.pt \
  --data-dir outputs/probe_data/other_eval_dataset \
  --split test
```

### SLURM End-to-End Probe Job

Submit one job that builds the probe dataset and trains the probe:

```bash
sbatch slurm/run_probe_train_e2e.sbatch
```

Default dataset/model settings in this job:
- `MODEL_PRESET=openthinker3_1p5b`
- `#SBATCH --gres=gpu:8` (job requests 8 GPUs by default)
- rollout `tp/dp` comes from `src/loop_probe/configs.py` preset defaults
- optional rollout concurrency override: `MAX_NUM_SEQS=...`
- optional completion-feature extraction throughput override: `COMPLETION_BATCH_SIZE=...` (default: `1`)
- `TRAIN_DATASET=HuggingFaceH4/MATH-500`, `TRAIN_SPLIT=test`
- `TEST_DATASET` omitted (defaults to local `data/aime_2024_2025.jsonl` in `build_probe_dataset.py`)
- `TEST_SPLIT=test`
- `PROMPT_FIELD=problem`
- `PROBE_PRESET=mlp`

Prompt formatting note:
- Probe dataset build and generation/eval scripts share the same chat prompt constructor: `scripts/utils.py:build_prompt()`.

## Model Presets

Predefined configurations for common models (see `src/loop_probe/configs.py`):

| Preset | Model | TP | DP | Temperature | Max Tokens |
|--------|-------|----|----|-------------|------------|
| `qwq_32b` | Qwen/QwQ-32B | 8 | 1 | 0.0 | 30000 |
| `openthinker3_7b` | open-thoughts/OpenThinker3-7B | 1 | 8 | 0.0 | 30000 |
| `openthinker3_1p5b` | open-thoughts/OpenThinker3-1.5B | 1 | 8 | 0.0 | 30000 |

Override any preset field via CLI:

```bash
--model-preset qwq_32b --temperature 0.3 --max-tokens 20000
```

Or skip presets entirely:

```bash
--model-id Qwen/QwQ-32B --tp 4 --temperature 0.0 --max-tokens 30000
```

## Loop Detection

A sequence is labeled as "looped" if any 30-gram appears ≥20 times in the generated token IDs. These defaults (`--loop-n 30`, `--loop-k 20`) can be adjusted during dataset building.

## Outputs

**Dataset build:**
- `{out_dir}/train/shard-*.pt` - Training shards (features + labels)
- `{out_dir}/test/shard-*.pt` - Test shards
- `{out_dir}/manifest.json` - Dataset metadata and configuration

**Training:**
- `{out_dir}/best.pt` - Best checkpoint (by ROC-AUC, then macro-F1)
- `{out_dir}/last.pt` - Final epoch checkpoint
- `{out_dir}/metrics.jsonl` - Per-epoch evaluation metrics
- `{out_dir}/best_metrics.json` - Best eval row summary for this run

**Multi-seed SLURM summary:**
- `${OUT_RUN_DIR}/seed_summary.json` - Per-seed best rows + aggregate mean/std
- `${OUT_RUN_DIR}/seed_summary.csv` - Aggregate mean/std table

**Multi-seed E2E training (`slurm/run_probe_train_e2e.sbatch`):**
- `{out_run_dir}/seed_*/metrics.jsonl` - Per-seed train/eval metrics
- `{out_run_dir}/probe_multiseed_curves.png` - Aggregated train/eval curves across seeds

Manual plotting command (if needed):
```bash
python scripts/plot_probe_multiseed.py \
  --run-dir outputs/probe_runs/<run_name> \
  --out outputs/probe_runs/<run_name>/probe_multiseed_curves.png
```

## Repository Structure

```
cot-loop/
├── src/loop_probe/          # Core probe training library
│   ├── configs.py           # Model presets and configuration
│   ├── hf_data.py           # Hugging Face dataset loading
│   ├── prefill.py           # Prefill activation extraction
│   ├── rollout.py           # vLLM trajectory generation
│   ├── labeling.py          # Loop detection and labeling
│   ├── dataloader.py        # PyTorch dataset/dataloader
│   ├── probes/              # Probe architectures
│   └── train_utils.py       # Training utilities
├── scripts/
│   ├── build_probe_dataset.py  # Extract features & labels
│   ├── train_probe.py          # Train probe classifier
│   ├── eval_probe_checkpoint.py # Evaluate torch probe checkpoints
│   ├── train_metadata_residual_probe.py # Metadata + stacked-feature residual probe
│   ├── aggregate_probe_runs.py # Multi-seed mean/std summary
│   └── [loop analysis scripts]
├── slurm/                   # SLURM batch scripts
├── data/                    # Input datasets
└── outputs/                 # Generated artifacts
```

## Advanced Usage

### Custom Model Configuration

```bash
python scripts/build_probe_dataset.py \
  --train-dataset my_org/prompts \
  --prompt-field text \
  --model-id meta-llama/Llama-3.1-8B \
  --tp 2 \
  --dtype float16 \
  --max-model-len 16384 \
  --temperature 0.0 \
  --max-tokens 10000 \
  --out-dir outputs/llama_probe
```

### Training Hyperparameters

```bash
python scripts/train_probe.py \
  --data-dir outputs/probe_data \
  --out-dir outputs/probe_runs/run1 \
  --wandb-project my-project \
  --probe-preset mlp \
  --epochs 20 \
  --batch-size 512 \
  --lr 1e-3 \
  --weight-decay 0.01 \
  --eval-every 2
```

### Class Imbalance

The training script automatically applies `pos_weight` in BCEWithLogitsLoss based on the train split's positive/negative ratio.

## Additional Scripts

For rollout-loop analysis on existing generations, see the `scripts/` directory.  
The Figure 1 plotting and generation scripts remain for archival comparisons and are not the primary detector training path.

## Technical Details

See [src/loop_probe/README.md](src/loop_probe/README.md) for detailed module documentation.

## Notes

- Prefill feature extraction uses Transformers and loads the full model on GPU
- Rollout generation uses vLLM for efficient batched inference
- Training requires `WANDB_API_KEY` in `.env` or environment
- GPU memory requirements depend on model size and batch size
