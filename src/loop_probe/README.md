# loop_probe

Purpose
- Build a binary probe dataset from LLM runs where:
  - input feature = last-token activation at last layer during prefill
  - target label = whether a rollout trajectory loops (`has_ngram_loop`)
- Train a simple linear classifier probe on those features.

High-level flow
1. Load Hugging Face dataset rows and read prompt text from `--prompt-field`.
2. Build train/test splits:
   - separate train/test dataset specs, or
   - one dataset split into train/test with deterministic random split.
3. Prefill pass (Transformers): extract one feature vector per prompt.
4. Rollout pass (vLLM): generate one trajectory per prompt.
5. Label with loop detector (`n`-gram repeated `k` times).
6. Save tensors as `.pt` shards and write `manifest.json`.
7. Train/eval a linear probe with weighted BCE and W&B logging.

Key modules
- `configs.py`: rollout config dataclass and model presets.
- `hf_data.py`: HF dataset loading and split utilities.
- `prefill.py`: hidden-state extraction for prefill features.
- `rollout.py`: rollout generation via vLLM.
- `labeling.py`: loop detector + label conversion.
- `serialization.py`: shard writing + manifest.
- `dataloader.py`: dataset/dataloader for training.
- `probes/linear_probe.py`: baseline linear classifier.
- `train_utils.py`: seed/device/metrics helpers.

Model presets
- `qwq_32b`: `Qwen/QwQ-32B`, `tp=8`, `dp=1`
- `openthinker3_7b`: `open-thoughts/OpenThinker3-7B`, `tp=1`, `dp=8`
- `openthinker3_1p5b`: `open-thoughts/OpenThinker3-1.5B`, `tp=1`, `dp=8`

You can override preset fields with CLI flags (`--model-id`, `--temperature`, `--max-tokens`, `--tp`, `--dp`, etc.).

## Usage

Build probe dataset
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

Single-dataset split (90/10 default)
```bash
python scripts/build_probe_dataset.py \
  --train-dataset my_org/my_dataset \
  --train-split train \
  --prompt-field prompt \
  --model-preset openthinker3_7b \
  --out-dir outputs/probe_data
```

Train probe
```bash
python scripts/train_probe.py \
  --data-dir outputs/probe_data \
  --out-dir outputs/probe_runs/run1 \
  --wandb-project cot-loop-probe
```

Outputs
- Dataset build:
  - `out_dir/train/shard-*.pt`
  - `out_dir/test/shard-*.pt`
  - `out_dir/manifest.json`
- Training:
  - `out_dir/best.pt`
  - `out_dir/last.pt`
  - `out_dir/metrics.jsonl`

Notes
- Training script loads `.env` and expects `WANDB_API_KEY`.
- Install deps with `uv sync` before running.
