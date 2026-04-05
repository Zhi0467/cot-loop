# Repository Guidelines

## Project Focus
This repository builds a chain-of-thought (CoT) failure predictor from configurable activation views plus the paired rollout-statistics line that explains where degenerate rollouts come from. The older target-choice debate is closed: regression `mean_relative_length` and binary `majority_s_0.5` remain the locked prompt-profile heads, and the balanced-train / natural-test regression rerun is now a finished artifact rather than an open request. In parallel, the OLMo degeneration-origin thread now has a corrected result surface: the fishy OLMo3 `MMLU-Pro` and `LiveCodeBench` rows have been repaired on the saved outputs, and the bounded OLMo2 `1B` ladder now serves as the cheap full-progression control showing base -> SFT -> RLVR cleanup of loop / max-length mass. Before launching work, read `roadmap.md`, `backlog.md`, `docs/prompt-profile-eval-contract.md`, `docs/prompt-profile-risk-screen-2026-03-30.md`, and `understand-where-loop-and-max-length-come-from.md`.

## Project Structure & Module Organization
- `src/loop_probe/`: Core library for prompt loading, prefill extraction, rollout generation, loop labeling, probe architectures, and training utilities.
- `scripts/`: CLI entry points for data/building, probe training, analysis, and plotting.
- `data/`: Local datasets and documentation. `data/README.md` defines the expected JSONL schema for non-HF local files.
- `slurm/`: SLURM batch scripts for probe pipeline, generation, and prefill-stability experiments.
- `outputs/`: Generated artifacts (prefill shards, checkpoints, metrics CSVs, figures).
- `roadmap.md`: chronological experiment log and milestone status.
- `backlog.md`: concrete next experiments and remaining measurement gaps.
- `pyproject.toml` + `uv.lock`: Dependency definitions (Python >= 3.10).
- `docs/prompt-profile-probe.md`: current prompt-profile implementation path and launch knobs.
- `docs/prompt-profile-eval-contract.md`: exact prompt-profile target, baseline, and metric definitions; read this before making claims about prompt-length controls or `Spearman`/capture metrics.

## Build, Test, and Development Commands
- Install dependencies: `uv sync`.
- Build probe dataset shards:
  `python scripts/build_probe_dataset.py --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> --model-preset openthinker3_1p5b --out-dir outputs/probe_data`.
- Build prompt-profile dataset shards:
  `python scripts/build_probe_dataset.py --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> --target-kind probability --num-generations 10 --profile-tail-threshold 0.9 --out-dir outputs/probe_data`.
- Build prompt-profile regression shards:
  `python scripts/build_probe_dataset.py --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> --target-kind regression --profile-target mean_relative_length --num-generations 10 --out-dir outputs/probe_data`.
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
