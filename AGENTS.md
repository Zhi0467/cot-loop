# Repository Guidelines

## Project Structure & Module Organization
- `scripts/`: Python entry points for dataset building, vLLM generation, metrics computation, and plotting.
- `slurm/`: SLURM batch entry points.
- `data/`: Input JSONL assets and documentation. `data/README.md` defines the expected AIME 2024/2025 format.
- `outputs/`: Generated artifacts (metrics CSVs, figures). This folder is created by scripts when needed.
- `FIG1_REPRO.md`: Step-by-step reproduction notes for Figure 1.
- `pyproject.toml` + `uv.lock`: Dependency definitions (Python >= 3.10).

## Build, Test, and Development Commands
- Install dependencies (recommended): `uv sync`.
- Build the dataset JSONL: `python scripts/build_aime_jsonl.py --out data/aime_2024_2025.jsonl`.
- Run vLLM generation (local):
  `python scripts/run_vllm_generate.py --model-id Qwen/QwQ-32B --data data/aime_2024_2025.jsonl --metrics-out outputs/qwq32b_metrics.csv --tp 8`.
- Run on SLURM: `sbatch slurm/run_vllm_generate.sbatch` (uses env vars like `MODEL_ID`, `TP`, `DP`).
- Compute metrics from existing generations: `python scripts/compute_metrics.py --generations path/to.jsonl --out outputs/metrics.csv`.
- Plot Figure 1: `python scripts/plot_fig1.py --metrics outputs/*_metrics.csv --out outputs/fig1.png`.

## Coding Style & Naming Conventions
- Python-only codebase; use 4-space indentation and keep functions small and single-purpose.
- Prefer explicit CLI args and type hints similar to existing scripts.
- Output naming: per-model metrics use `*_metrics.csv`; figures live in `outputs/`.
- No formatter/linter is configured; follow PEP 8 conventions for consistency.

## Testing Guidelines
- No automated test suite is present.
- Use the “Smoke tests” in `scripts/run_vllm_generate.py` as quick validation (e.g., `TEMPS=0`, small `N`, short `MAX_TOKENS`).
- Validate dataset structure against `data/README.md` before long runs.

## Commit & Pull Request Guidelines
- Git history uses short, lowercase, imperative-style messages (e.g., "added DP", "fixed vLLM config under greedy"). Keep messages concise.
- PRs should include a brief purpose, exact reproduction command(s), hardware/GPUs used, and expected artifacts (CSV/PNG). Link any new datasets or model variants introduced.
