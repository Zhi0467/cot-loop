# CoT Loop Detection via Probe Classifiers

This repository studies whether chain-of-thought loop risk is predictable from internal activations and rollout telemetry. The original workflow centered on prompt-prefill last-token probes, and the current active stage is a prompt-profile RFM-plus-steering extension on the frozen Qwen prompt-profile bundle. The retained collaborator-facing benchmark set for that stage is `GPQA`, `MATH-500`, `MMLU-Pro`, and `LiveCodeBench`; `AIME` is intentionally out on this object because it mostly reads as a prompt-visible workload case. The merged trigger-attention audit remains useful background context, but it is no longer the live blocker surface for the next stage.

## Overview

The project now has three durable evidence streams:
- the probe line asks whether loop risk is detectable from stacked prompt-prefill activations, either by slicing one layer or by voting across all layers;
- the rollout-statistics line measures how often looping and max-length hits actually occur under the repaired v2 collector contract across `MATH-500`, `AIME`, `GPQA`, `MMLU-Pro`, and `LiveCodeBench`;
- the trigger-attention line replays saved loop rows to ask where the model is attending around the repeated trigger region, but that line should now be treated as merged background evidence rather than the active phase boundary.

Latest status:
- upstream PRs `#9` and `#10` are both merged, so the repo is no longer sitting on an open GitHub review surface for the old prompt-profile or trigger-attention lines.
- the merged trigger-attention note is still scientifically narrow:
  - it is background evidence about prompt-dominant final-layer attention plus a mid-stack previous-loop signal on the saved loop rows;
  - it is not validation for the next-stage RFM steering object.
- the collaborator-facing prompt-profile surface is now the unified note `docs/prompt-profile-unified-report-2026-04-09.md` plus `outputs/prompt_profile_unified_report_20260409/`, which folds the April 5 combined audit, the April 6 mechanism note, and the April 9 plain-English length audit into one canonical PDF while keeping the natural-regression lane, the balanced-binary recommendation, and the prompt-shape mechanism answer in the same surface.
- the next execution surface is now pinned in `docs/prompt-profile-rfm-steering-plan-2026-04-21.md`:
  - reuse the saved March `2026-03-22` / `2026-03-23` Qwen prompt-profile archives;
  - keep the retained four-benchmark set `GPQA`, `MATH-500`, `MMLU-Pro`, and `LiveCodeBench`, with `AIME` intentionally excluded from the collaborator-facing stage;
  - keep the binary head `majority_s_0.5`;
  - add a native layerwise RFM path as a sibling baseline to the current activation linear and activation MLP surfaces;
  - extend the report with direction-coherence diagnostics before making any steering claim;
  - keep `T = 5` fixed on the first RFM pass;
  - then run paired benchmark-local spherical steering at fixed `t = 0.3` using the exported per-layer bundle directly, not a top-`k` rule or controller;
  - include `no_steer`, `-v`, `+v`, and random-direction controls in the first steering table;
  - then test one external benchmark with the averaged "verbose" vector rather than doing leave-one-benchmark-out gymnastics inside the retained training set.
- the repo now has committed stage-0 RFM scaffolding:
  - `src/loop_probe/prompt_profile_rfm_stage_registry.py`
  - `scripts/emit_prompt_profile_rfm_stage_registry.py`
  - `scripts/validate_prompt_profile_rfm_stage_registry.py`
  - `src/loop_probe/stage_artifacts.py`
  - `outputs/prompt_profile_rfm_stage0_registry_validation_20260421/registry_validation.json`
- the repo now has a live native RFM detector path:
  - `src/loop_probe/rfm.py`
  - `scripts/train_prompt_profile_rfm.py`
  - `slurm/run_prompt_profile_rfm.sbatch`
- the repo now has a live vector-export surface:
  - `scripts/export_prompt_profile_rfm_vectors.py`
- one important repo-reality correction changed the current detector object:
  - the saved March prompt-profile archives were built with `archive_tail_threshold = 0.9`
  - this stage is about `majority_s_0.5`
  - so the honest stage label is now recomputed from saved rollout `relative_length` rather than trusted from the archive's saved `0.9` label
- the repaired `LiveCodeBench` detector/vector artifact now exists on current project head `120a808`:
  - output root: `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_rfm/livecodebench_full_bootstrap200_seed0_metricfix_20260421/`
  - repaired split: fit-train / val / test counts `280 / 128 / 160`, positives `140 / 35 / 54`
  - best validation layer/bandwidth: `27 / 100`
  - validation `PR-AUC`: `0.6555`
  - test `PR-AUC`: `0.7055`
  - test `ROC-AUC`: `0.8590`
  - test positive `F1`: `0.2222`
- the earlier `54 / 128 / 160` `LiveCodeBench` comparison object is now superseded and should not be reused for the stage report
- repaired prompt-only baseline bundle:
  - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_prompt_baselines/livecodebench_majority_s0p5_rolloutrecompute_seed0_20260421/`
  - `prompt_length`: test `PR-AUC 0.5771`, test `ROC-AUC 0.7201`
  - `prompt_shape_linear`: test `PR-AUC 0.5871`, test `ROC-AUC 0.7290`
- repaired activation baseline bundle:
  - `/data/scratch/murphy/outputs/cot-loop-detection/prompt_profile_stage_baselines/livecodebench_majority_s0p5_rolloutrecompute_seed0_20260421/`
  - `best_rank` mean test `PR-AUC`: linear last-layer `0.4163`, linear ensemble `0.5698`, `mlp256d1` last-layer `0.7055`, `mlp256d1` ensemble `0.6637`
  - `best_loss` mean test `PR-AUC`: `mlp256d1` last-layer `0.7147`
- current honest repaired detector read:
  - RFM is above the cheap prompt-only baselines and above activation linear on the repaired split
  - `h256 d1` MLP last-layer is now essentially tied with the current single-seed RFM on `PR-AUC`, and it is slightly ahead if the detector comparison uses the activation `best_loss` mean
  - because the repaired activation bundle is multiseed while RFM is still single-seed, the next honest detector question is whether RFM also needs matching multiseed / split-seed sweeps before the report is locked
- the repaired LiveCodeBench vector bundle is now a real stage-2 artifact, not just an export stub:
  - `vector_exports/summary.json` carries signed per-layer vectors, prompt-ID provenance, raw / normalized checksums, held-out 1D projection separation, and cross-layer cosine structure
  - `2026-04-21` direction-bootstrap replay now adds `100` fixed-hyperparameter bootstrap refits per layer under the same sign convention
  - all `28` layers clear mean cosine `0.781` or better against their exported reference direction, and the weakest `95%` low bound is still `0.693`
  - late layers `23-26` are the most direction-stable (`0.867` to `0.909` mean cosine), while validation selection still peaks at layer `27`
  - cross-benchmark cosine alignment is still missing before any transfer or averaged-vector claim
- the first `LiveCodeBench` full run also forced one implementation correction:
  - the original `cholesky` default failed on a non-positive-definite kernel matrix;
  - the committed default solver is now `solve`.
- the repo already has activation-side linear controls distinct from the prompt-only metadata baselines:
  - `src/loop_probe/probes/linear_probe.py`
  - `scripts/run_prompt_profile_full_train.py`
  - `docs/prompt-profile-unified-report-2026-04-09.md`
- the common-policy rollout-statistics bundle remains refreshed under one shared decode policy (`temperature=0.2`, `num_generations=10`, and `max prompts <= 800` where applicable) across `MATH-500`, `AIME`, `GPQA`, capped `MMLU-Pro`, and capped `LiveCodeBench release_v6`.
- the repaired MC rows are materially different from the stale pre-refresh bundle: `GPQA` now reports `34.5%` rollout success instead of `3.1%`, and `MMLU-Pro` now reports `65.2%` instead of `14.2%`, both under the terminal JSON-answer contract.
- the canonical regression lane is now pinned twice over: the original locked April natural-split / natural-sampler `mean_relative_length` run plus Slurm `2215`, which reproduced that ledger exactly from the current branch.
- binary `majority_s_0.5` remains the cleaner deployment-facing head, and the current best single global activation surface on that balanced-binary object is still `ensemble h256 d1`.
- the current target split is explicit and stable:
  - regression keeps the natural prompt-disjoint train/test split with natural sampling;
  - balanced training is only for the binary `majority_s_0.5` head.
- the OLMo degeneration line is now on a corrected result surface rather than an unresolved warning:
  - `RLVR / MMLU-Pro = 0 / 80` was a grader bug on relaxed terminal JSON-like forms such as `{"answer": I}`;
  - OLMo now has a model-native `LiveCodeBench` adapter path instead of the old silent Qwen-wrapper fallback;
- the cheap full-ladder control is now the finished OLMo 2 `1B` 50-prompt progression, which shows heavy loop / cap mass in base, reduced but still present mass in SFT, and much smaller mass in `RLVR1` / instruct.
- that OLMo2 ladder now has a proper visualization bundle too:
  - progression figures and Sankey/alluvial overlap views live under `outputs/olmo2_1b_progression_bound50_20260406/`
  - those figures are built directly from the saved `50`-prompt stage JSONs, not hand-copied tables.
- the same-family Qwen follow-up is now finished rather than only active:
  - the old Qwen reference object is the repaired v2 rollout bundle `outputs/qwen3_1p7b_rollout_stats_v2_temp0p2_gen10/`
  - the matching base-control rerun uses `Qwen/Qwen3-1.7B-Base` on the same sampler and dataset family, but at the base checkpoint's real `32768` context limit rather than the instruct checkpoint's `40960`
  - the `LiveCodeBench` comparison is only same dataset plus same sampler, not a literal same-LM-style replay: the old v2 instruct row used `CodeQwenInstruct`, while the base control uses `GenericBase`
  - the finished base bundle itself is heavily degenerate across all five datasets, with `AIME 114/500` looped, `GPQA 190/500`, `MMLU-Pro 157/500`, and `LiveCodeBench 232/500`
  - the saved text probe shows that Qwen base raw does degenerate on MCQ, but mainly by repeating the answer-format instruction tail rather than by OLMo-style math-derivation loops
  - the saved instruct-side v2 table is still useful as rough scale, but not as a controlled per-dataset comparison, because prompt pools, rollout counts, context limits, and `LiveCodeBench` LM style do not match the new base bundle
- the current open work is now:
  - decide whether the repaired detector lane is ready to freeze on the current single-seed RFM row, or whether RFM itself also needs matching multiseed / split-seed sweeps;
  - extend the unified prompt-profile report with the repaired detector table plus direction-coherence diagnostics on that repaired surface;
  - keep exporting repaired benchmark-local steering vectors and run paired fixed-`t = 0.3` spherical steering tables even if RFM does not become the single best detector;
  - then test one external benchmark with the averaged "verbose" vector;
  - then revisit stronger prompt-shape controls and only later ablate layer rules, controllers, or `t` if the first steering table shows signal.

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
