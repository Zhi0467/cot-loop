# loop_probe

Purpose
- Build a binary probe dataset from LLM runs to train a CoT loop detector.
- The canonical dataset stores the last-token activation from every transformer layer as a stacked `[layer, hidden]` tensor per prompt.
- Train a probe classifier (linear or MLP) either on one selected layer or as a layerwise voting ensemble.

High-level flow
1. Load Hugging Face dataset rows and read prompt text from `--prompt-field`.
2. Build model-formatted chat prompts with `utils.build_prompt` (shared by data builders and analysis scripts).
3. Build train/test splits:
   - separate train/test dataset specs, or
   - identical train/test specs split deterministically with `--split-ratio`.
4. Prefill pass (Transformers): extract last-token activations from every layer and store them as a stacked tensor `[layer, hidden]` per prompt.
5. Rollout pass (vLLM): generate one trajectory per prompt.
6. Label with loop detector (`n`-gram repeated `k` times).
7. Save tensors as `.pt` shards and write `manifest.json`.
8. Train/eval a probe classifier with weighted BCE and W&B logging.

Key modules
- `configs.py`: rollout + probe config dataclasses, presets, and probe model factory.
- `hf_data.py`: HF dataset loading and split utilities.
- `prefill.py`: hidden-state extraction for prefill features.
- `rollout.py`: rollout generation via vLLM.
- `labeling.py`: loop detector + label conversion.
- `serialization.py`: shard writing + manifest.
- `dataloader.py`: dataset/dataloader for training.
- `probes/linear_probe.py`: baseline linear classifier.
- `probes/mlp_probe.py`: one-hidden-layer MLP classifier.
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

The default `build_probe_dataset.py` feature view is
`last_token_all_layers_stack_final`, a stacked `[layer, hidden]` tensor for the
last token at every layer.

Build the default stacked dataset:
```bash
python scripts/build_probe_dataset.py \
  --train-dataset HuggingFaceH4/MATH-500 \
  --train-split test \
  --prompt-field problem \
  --model-preset openthinker3_1p5b \
  --out-dir outputs/probe_data/layers_stack
```

Omit `--test-dataset` to use local `data/aime_2024_2025.jsonl` as test set
```bash
python scripts/build_probe_dataset.py \
  --train-dataset my_org/my_dataset \
  --train-split train \
  --prompt-field prompt \
  --model-preset openthinker3_7b \
  --out-dir outputs/probe_data
```

For this default local test file, prompt loading is hardcoded to `question`
and requires an `answer` field on each row.

Train probe
```bash
python scripts/train_probe.py \
  --data-dir outputs/probe_data \
  --out-dir outputs/probe_runs/run1 \
  --wandb-project cot-loop-probe
```

Default training uses an MLP in `last_layer` mode and slices the final layer
from a stacked `[layer, hidden]` feature view.

Train an ensemble over all layers
```bash
python scripts/train_probe.py \
  --data-dir outputs/probe_data \
  --out-dir outputs/probe_runs/run1_ensemble \
  --classifier-mode ensemble \
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
  - `metrics.jsonl` rows include eval metrics:
    `accuracy`, `macro_f1`, `roc_auc`, `pr_auc`,
    `positive_precision`, `positive_recall`, `positive_f1`, `prevalence`

Notes
- Training script loads `.env` and expects `WANDB_API_KEY`.
- `train_probe.py` supports optional MLP overrides:
  `--mlp-hidden-dim`, `--mlp-depth`, `--mlp-dropout`.
- Ensemble mode uses per-layer hard votes, with ties resolving negative.
- Install deps with `uv sync` before running.
