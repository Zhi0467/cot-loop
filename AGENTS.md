# Repository Guidelines

## Project Focus
This repository builds a chain-of-thought (CoT) failure predictor from configurable activation views plus the paired rollout-statistics line that explains where degenerate rollouts come from. The active execution object is now the corrected four-dataset rollout-stat rebuild documented in `docs/weeks/2026-W17/main-four-dataset-rollout-rebuild-2026-04-23.md`: `LiveCodeBench`, `TACO-hard`, `MATH level-5`, and `Omni-MATH >= 7`, each collected with explicit thinking `on` / `off` surfaces. Every run on a dataset produces exactly one `rollout_bundle.v1` pair (`<base>.jsonl.gz` for full per-prompt replay, `<base>.json` for metadata + aggregate stats; see `docs/reference/rollout-bundle-v1-schema.md`) so the same rollouts drive prompt-profile relabeling, probe training, and steering without re-running the GPU work. The sizing rule is no longer the earlier `800`-prompt slice: use the full dataset when it is under `1000`, otherwise use `1000`, except retained `LiveCodeBench` `release_v6`, which intentionally uses the full `1055` prompts. The older March-repair, LiveCodeBench-only rerun, and mistaken `LiveCodeBench-extra` branch are historical debugging context, not the current stage contract. Before launching work, read `roadmap.md`, `backlog.md`, `docs/weeks/2026-W17/main-four-dataset-rollout-rebuild-2026-04-23.md`, `docs/weeks/2026-W13/prompt-profile-eval-contract.md`, and `docs/weeks/2026-W14/understand-where-loop-and-max-length-come-from.md`.

## Standing TODOs
- `docs/standing/todos-2026-W17.md` is the active standing TODO ledger for the current rebuild.
- When given a new task that changes active project work, update the standing TODO entry before or while starting the task.
- When a task is completed, paused, or its status changes, update the same standing TODO entry so the ledger reflects the current state.
- Keep standing TODOs simple: track the current active workstreams, not historical notes.

## Rollout Artifact Hygiene
- Delete stale failed launch artifacts after recording the failure in the standing TODO or relevant run note. "Stale failed launch" means a job/dry-run/canceled duplicate that failed before producing real prompt rows or a valid `rollout_bundle.v1` pair, for example bad sbatch/env setup, missing cache/checkout, failed conda activation, or launcher dry-run manifests.
- Do not treat mid-run partials the same way. If a rollout fails after generating some prompts, preserve its partial bundle/logs/checkpoints for diagnosis or resume unless the user explicitly asks to remove them. Partial data can explain the failure mode and may be recoverable; startup-stale artifacts only add queue/log clutter.
- Before deleting, verify the current queue and active output root so running jobs and the latest intended manifest/logs are not removed.

## Rollout CPU Finalization
- Do not run CPU-only post-hoc grading inside GPU Slurm allocations. `livecodebench_codegen` and `taco_codegen` collection jobs should defer post-hoc grading with `DEFER_CPU_FINALIZE=1`, exit after GPU generation, and write a deferred sidecar.
- Finalize deferred LCB/TACO bundles outside Slurm with GPUs hidden, for example `CUDA_VISIBLE_DEVICES='' uv run python scripts/rollout/collect_model_stats.py --task-kind livecodebench_codegen --out <bundle-base>.json --livecodebench-repo <repo> --finalize-only`.
- If a deferred GPU job is canceled after writing rows, resume with `RESUME=1`; the collector can reuse both final bundles and `__rank*.jsonl.gz` partials.

## Project Structure & Module Organization
- `configs/rollout/main_rollout_stats_suite.json`: Canonical four-dataset paired thinking `on` / `off` rollout-stats rebuild contract, including shared sampling settings and per-dataset surfaces.
- `src/probe/`: Core library for prompt loading, prefill extraction, rollout generation, loop labeling, probe architectures, and training utilities. `main_rollout_stats_suite.py` loads the canonical rollout config and translates it into collector environments; `collector.py` is the repaired v2 rollout-statistics path it drives.
- `src/steer/`: Steering-stage registry and artifact helpers shared by RFM/vector export/steering CLIs.
- `scripts/`: CLI entry points for data/building, probe training, analysis, and plotting. `scripts/rollout/launch_main_rollout_stats_suite.py` is the active sbatch wrapper for the rebuild and propagates `CONDA_ENV` / `CONDA_DEFAULT_ENV` plus `THINKING_MODE` into submitted jobs.
- `data/`: Local datasets and documentation. `data/README.md` defines the expected JSONL schema for non-HF local files.
- `slurm/`: SLURM batch scripts grouped into `rollout/`, `train/`, `mechanism_analysis/`, and `steer/`.
- `outputs/`: Generated artifacts (prefill shards, checkpoints, metrics CSVs, figures).
- `roadmap.md`: chronological experiment log and milestone status.
- `backlog.md`: concrete next experiments and remaining measurement gaps.
- `pyproject.toml` + `uv.lock`: Dependency definitions (Python >= 3.10).
- `docs/weeks/2026-W14/prompt-profile-probe.md`: current prompt-profile implementation path and launch knobs.
- `docs/weeks/2026-W13/prompt-profile-eval-contract.md`: exact prompt-profile target, baseline, and metric definitions; read this before making claims about prompt-length controls or `Spearman`/capture metrics.

## Build, Test, and Development Commands
- Install dependencies: `uv sync`.
- Build probe dataset shards:
  `python scripts/data/build_probe_dataset.py --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> --model-preset openthinker3_1p5b --out-dir outputs/probe_data`.
- Build prompt-profile dataset shards:
  `python scripts/data/build_probe_dataset.py --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> --target-kind probability --num-generations 10 --profile-tail-threshold 0.9 --out-dir outputs/probe_data`.
- Build prompt-profile regression shards:
  `python scripts/data/build_probe_dataset.py --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> --target-kind regression --profile-target mean_relative_length --num-generations 10 --out-dir outputs/probe_data`.
- Train probe:
  `python scripts/train/train_probe.py --data-dir outputs/probe_data --out-dir outputs/probe_runs/run1 --probe-preset linear --wandb-project cot-loop-probe`.
- Train prompt-profile probe:
  `python scripts/train/train_probe.py --data-dir outputs/probe_data --out-dir outputs/probe_runs/run1 --probe-preset mlp --classifier-mode ensemble --score-rule mean_prob --wandb-project cot-loop-probe`.
- Run probe end-to-end on SLURM:
  `sbatch slurm/train/run_probe_train_e2e.sbatch`.
- Launch the current canonical four-dataset rollout-stats rebuild (paired thinking on/off, `Qwen/Qwen3-1.7B`):
  `python scripts/rollout/launch_main_rollout_stats_suite.py --output-root outputs/model_stats/main_rollout_stats_rebuild --thinking-modes on,off --submit`.
  The suite definition (model, sampling config, per-dataset contracts) lives in `configs/rollout/main_rollout_stats_suite.json`; do not redefine sampling or the dataset list ad hoc.
- Generate rollout data for older standalone labeling/analysis (not used by the current rebuild):
  `python scripts/rollout/run_vllm_generate.py --model-id Qwen/QwQ-32B --data data/aime_2024_2025.jsonl --metrics-out outputs/qwq32b_metrics.csv --tp 8`.
- Run on SLURM for that older generation path:
  `sbatch slurm/rollout/run_vllm_generate.sbatch`.
- Summarize loop metrics from generations:
  `python scripts/rollout/compute_metrics.py --generations path/to.jsonl --out outputs/metrics.csv`.
- Run plotting scripts as needed:
  `python scripts/plot/plot_accuracy_vs_temperature.py --metrics outputs/*_metrics.csv --out outputs/accuracy_plot.png`.

## Coding Style & Naming Conventions
- Python-only codebase; use 4-space indentation and keep functions small and single-purpose.
- Prefer explicit CLI args and type hints similar to existing scripts.
- Outputs keep detector naming: checkpoints live in run folders, metrics in `_metrics.csv`, and figures in `outputs/`.
- No formatter/linter is configured; follow PEP 8 conventions for consistency.

## Testing Guidelines
- No automated test suite is present.
- Use small-scale local checks for smoke validation (small `--train-max-samples`, `--test-max-samples`, and short rollout limits in generation scripts) before long jobs.
- Validate local data schema with `data/README.md` and `scripts/rollout/run_vllm_generate.py`/`scripts/data/build_probe_dataset.py` behavior.

## Commit & Pull Request Guidelines
- Git history uses short, lowercase, imperative-style messages (e.g., `added dp`, `fixed probe batching`). Keep messages concise.
- PRs should include a brief purpose, exact reproduction command(s), and expected artifacts (checkpoints/metrics/plots).
