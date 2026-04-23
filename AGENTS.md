# Repository Guidelines

## Project Focus
This repository builds a chain-of-thought (CoT) failure predictor from configurable activation views plus the paired rollout-statistics line that explains where degenerate rollouts come from. The active execution object is now the corrected four-dataset rollout-stat rebuild documented in `docs/main-four-dataset-rollout-rebuild-2026-04-23.md`: `LiveCodeBench`, `TACO-hard`, `MATH level-5`, and `Omni-MATH >= 7`, each collected with explicit thinking `on` / `off` surfaces and reusable prompt-rollout archives for later prompt-profile relabeling, probe training, and steering. The active sizing rule is the suite contract, not the earlier `800`-prompt slice: `LiveCodeBench` uses the full `release_v6` surface (`1055`), `TACO-hard` uses `1000 / 5536`, `MATH level-5` uses `1000 / 2304`, and `Omni-MATH >= 7` uses the full `916`-row HF slice. The older March-repair, LiveCodeBench-only rerun, and mistaken `LiveCodeBench-extra` branch are historical debugging context, not the current stage contract. Before launching work, read `roadmap.md`, `backlog.md`, `docs/main-four-dataset-rollout-rebuild-2026-04-23.md`, `docs/prompt-profile-eval-contract.md`, and `understand-where-loop-and-max-length-come-from.md`.

## Project Structure & Module Organization
- `src/loop_probe/`: Core library for prompt loading, prefill extraction, rollout generation, loop labeling, probe architectures, and training utilities. `main_rollout_stats_suite.py` holds the canonical four-dataset paired thinking `on` / `off` rebuild contract; `collector.py` is the repaired v2 rollout-statistics path it drives.
- `scripts/`: CLI entry points for data/building, probe training, analysis, and plotting. `launch_main_rollout_stats_suite.py` is the active sbatch wrapper for the rebuild and propagates `CONDA_ENV` / `CONDA_DEFAULT_ENV` plus `THINKING_MODE` into submitted jobs.
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
- Launch the current canonical four-dataset rollout-stats rebuild (paired thinking on/off, `Qwen/Qwen3-1.7B`):
  `python scripts/launch_main_rollout_stats_suite.py --output-root outputs/model_stats/main_rollout_stats_rebuild --thinking-modes on,off --submit`.
  The suite definition (model, sampling config, per-dataset contracts) lives in `src/loop_probe/main_rollout_stats_suite.py`; do not redefine sampling or the dataset list ad hoc.
- Generate rollout data for older standalone labeling/analysis (not used by the current rebuild):
  `python scripts/run_vllm_generate.py --model-id Qwen/QwQ-32B --data data/aime_2024_2025.jsonl --metrics-out outputs/qwq32b_metrics.csv --tp 8`.
- Run on SLURM for that older generation path:
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
