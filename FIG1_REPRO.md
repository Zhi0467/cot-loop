Figure 1 minimal reproduction (vLLM)

Overview
- Models: `Qwen/QwQ-32B`, `open-thoughts/OpenThinker3-7B`, `open-thoughts/OpenThinker3-1.5B`
- Dataset: AIME 2024 I/II + AIME 2025 I/II (60 problems)
- Temperatures: 0, 0.2, 0.4, 0.6, 0.8, 1.0
- Samples: 20 per (problem, model, temperature)
- Max tokens: 30000
- Looping: any 30-gram in model token space appears >= 20 times

1) Prepare dataset
- Create `data/aime_2024_2025.jsonl` (see `data/README.md`).

2) Generate responses + metrics (vLLM)
Run each model separately. On an 8-GPU node, use all GPUs for QwQ-32B and single GPU for the smaller models.
Metrics are computed on the fly using model token IDs (30-gram >= 20).

QwQ-32B (8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/run_vllm_generate.py \
  --model-id Qwen/QwQ-32B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/qwq32b_metrics.csv \
  --no-rollouts \
  --tp 8

OpenThinker3-7B (1 GPU)
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_vllm_generate.py \
  --model-id open-thoughts/OpenThinker3-7B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/openthinker3_7b_metrics.csv \
  --no-rollouts \
  --tp 1

OpenThinker3-1.5B (1 GPU)
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_vllm_generate.py \
  --model-id open-thoughts/OpenThinker3-1.5B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/openthinker3_1p5b_metrics.csv \
  --no-rollouts \
  --tp 1

Notes
- `run_vllm_generate.py` uses `tokenizer.apply_chat_template` and the prompt:
  {question}
  Please reason step by step, and put your final answer within \boxed{}.
- It pulls `top_p` and `top_k` from each model's HF GenerationConfig and sets `repetition_penalty=1.0`.
- If a model rejects `--max-model-len 32768`, lower it or omit the flag.
- To keep rollouts for inspection, remove `--no-rollouts` and set `--out`; you can also delete rollouts after metrics with `--delete-rollouts`.

3) Combine metrics
Concatenate the three per-model metrics CSVs.

python - <<'PY'
import csv
import glob

rows = []
for path in glob.glob("outputs/*_metrics.csv"):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(list(reader))

with open("outputs/fig1_metrics.csv", "w", encoding="utf-8", newline="") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
PY

4) Plot (optional)
python scripts/plot_fig1.py \
  --metrics outputs/fig1_metrics.csv \
  --out outputs/fig1.png

Expected outputs
- `outputs/fig1_metrics.csv` with columns: model_id, temperature, num_samples, loop_fraction, avg_tokens
- `outputs/fig1.png` (two-panel plot similar to Figure 1)
