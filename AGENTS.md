# Repository Guidelines

## Project Focus
This repository builds a chain-of-thought (CoT) failure predictor from configurable activation views. The current focus is no longer just binary loop labeling: after the repaired cross-dataset rollout-statistics refresh, the cap-generalization follow-up, and the Athena cross-check, the live recommendation is to move the next prefill round toward an in-distribution prompt-level rollout-profile predictor centered on normalized-length tail risk plus dense normalized-length regression, while keeping raw max-length-hit risk as a fixed-policy operational eval head rather than the sole main objective. If only one head ships first, the fallback target is now `s_0.9 = P(L / E >= 0.9)`, not `mu_log_rel`, and the first GPQA pass should keep the effective budget `E` controlled tightly enough that the target does not mostly reduce to prompt length. That first single-head path now exists in-repo for prefill features: use `build_probe_dataset.py --target-kind probability --num-generations 10 --profile-tail-threshold 0.9` and train it with `train_probe.py`, optionally `--classifier-mode ensemble --score-rule mean_prob`. The recovered five-dataset rollout bundle remains the empirical reference surface. The explicit `LiveCodeBench` caveat is unchanged: the crashed run still leaves `avg_first_loop_prefix_length` irrecoverable even though the correctness / loop / max-length / native `pass@k` block was recovered.

## Project Structure & Module Organization
- `src/loop_probe/`: Core library for prompt loading, prefill extraction, rollout generation, loop labeling, probe architectures, and training utilities.
- `scripts/`: CLI entry points for data/building, probe training, analysis, and plotting.
- `data/`: Local datasets and documentation. `data/README.md` defines the expected JSONL schema for non-HF local files.
- `slurm/`: SLURM batch scripts for probe pipeline, generation, and prefill-stability experiments.
- `outputs/`: Generated artifacts (prefill shards, checkpoints, metrics CSVs, figures).
- `pyproject.toml` + `uv.lock`: Dependency definitions (Python >= 3.10).
- `docs/prompt-profile-probe.md`: current prompt-level `s_0.9` implementation path and launch knobs.

## Build, Test, and Development Commands
- Install dependencies: `uv sync`.
- Build probe dataset shards:
  `python scripts/build_probe_dataset.py --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> --model-preset openthinker3_1p5b --out-dir outputs/probe_data`.
- Build prompt-profile dataset shards:
  `python scripts/build_probe_dataset.py --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> --target-kind probability --num-generations 10 --profile-tail-threshold 0.9 --out-dir outputs/probe_data`.
- Train probe:
  `python scripts/train_probe.py --data-dir outputs/probe_data --out-dir outputs/probe_runs/run1 --probe-preset linear --wandb-project cot-loop-probe`.
- Train prompt-profile probe:
  `python scripts/train_probe.py --data-dir outputs/probe_data --out-dir outputs/probe_runs/run1 --probe-preset mlp --classifier-mode ensemble --score-rule mean_prob --wandb-project cot-loop-probe`.
- Run probe end-to-end on SLURM:
  `sbatch slurm/run_probe_train_e2e.sbatch`.
- Generate rollout data for labeling/analysis (optional):
  `python scripts/run_vllm_generate.py --model-id Qwen/QwQ-32B --data data/aime_2024_2025.jsonl --metrics-out outputs/qwq32b_metrics.csv --tp 8`.
- Run on SLURM for generation:
  `sbatch slurm/run_vllm_generate.sbatch`.
- Summarize loop metrics from generations:
  `python scripts/compute_metrics.py --generations path/to.jsonl --out outputs/metrics.csv`.
- Run plotting scripts as needed:
  `python scripts/plot_accuracy_vs_temperature.py --metrics outputs/*_metrics.csv --out outputs/accuracy_plot.png`.

## Coding Style & Naming Conventions
- Python-only codebase; use 4-space indentation and keep functions small and single-purpose.
- Prefer explicit CLI args and type hints similar to existing scripts.
- Outputs keep detector naming: checkpoints live in run folders, metrics in `_metrics.csv`, and figures in `outputs/`.
- No formatter/linter is configured; follow PEP 8 conventions for consistency.

## Testing Guidelines
- No automated test suite is present.
- Use small-scale local checks for smoke validation (small `--train-max-samples`, `--test-max-samples`, and short rollout limits in generation scripts) before long jobs.
- Validate local data schema with `data/README.md` and `scripts/run_vllm_generate.py`/`scripts/build_probe_dataset.py` behavior.

## Commit & Pull Request Guidelines
- Git history uses short, lowercase, imperative-style messages (e.g., `added dp`, `fixed probe batching`). Keep messages concise.
- PRs should include a brief purpose, exact reproduction command(s), and expected artifacts (checkpoints/metrics/plots).
