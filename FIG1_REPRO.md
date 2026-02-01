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
  --metrics-out outputs/qwq32b_metrics.rep1.csv \
  --tp 8

OpenThinker3-7B (1 GPU)
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_vllm_generate.py \
  --model-id open-thoughts/OpenThinker3-7B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/openthinker3_7b_metrics.rep1.csv \
  --tp 1

OpenThinker3-1.5B (1 GPU)
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_vllm_generate.py \
  --model-id open-thoughts/OpenThinker3-1.5B \
  --data data/aime_2024_2025.jsonl \
  --metrics-out outputs/openthinker3_1p5b_metrics.rep1.csv \
  --tp 1

Notes
- `run_vllm_generate.py` uses `tokenizer.apply_chat_template` and the prompt:
  {question}
  Please reason step by step, and put your final answer within \boxed{}.
- It pulls `top_p` and `top_k` from each model's HF GenerationConfig and sets `repetition_penalty=1.0`.
- If a model rejects `--max-model-len 32768`, lower it or omit the flag.

3) Plot (optional)
Pass the three per-model metrics CSVs directly (no rollout files needed).

python scripts/plot_fig1.py \
  --metrics outputs/qwq32b_metrics.rep1.csv \
  --metrics outputs/openthinker3_7b_metrics.rep1.csv \
  --metrics outputs/openthinker3_1p5b_metrics.rep1.csv \
  --out outputs/fig1.png

Alternative: glob the metrics files or pass the output directory.

python scripts/plot_fig1.py --metrics "outputs/*_metrics*.csv" --out outputs/fig1.png
python scripts/plot_fig1.py --metrics outputs --out outputs/fig1.png

Expected outputs
- `outputs/qwq32b_metrics.rep1.csv`, `outputs/openthinker3_7b_metrics.rep1.csv`,
  `outputs/openthinker3_1p5b_metrics.rep1.csv` (per-model metrics with columns:
  model_id, temperature, num_samples, loop_fraction, avg_tokens)
- `outputs/fig1.png` (two-panel plot similar to Figure 1)
