#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from probe.main_rollout_stats_suite import (  # noqa: E402
    DEFAULT_SUITE_CONFIG_PATH,
    build_collect_env,
    get_suite_dataset,
    load_suite_definition,
    suite_dataset_keys,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_SUITE_CONFIG_PATH,
        help="JSON rollout suite config to load.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated suite dataset keys. Defaults to all datasets in --config.",
    )
    parser.add_argument(
        "--thinking-modes",
        default="on,off",
        help="Comma-separated thinking modes to launch (on/off).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the suite jobs with sbatch. Default is print-only.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Set RESUME=1 for submitted collector jobs so partial bundles are reused.",
    )
    parser.add_argument(
        "--job-prefix",
        default="q3-main",
        help="Prefix for sbatch job names.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(ROOT, "outputs", "model_stats", "main_rollout_stats_rebuild"),
        help="Root directory for collector JSONs and the suite manifest.",
    )
    parser.add_argument(
        "--manifest-name",
        default="suite_manifest.json",
        help="Manifest filename to write under --output-root.",
    )
    parser.add_argument(
        "--slurm-script",
        default=os.path.join(ROOT, "slurm", "rollout", "run_collect_model_stats.sbatch"),
        help="Collector sbatch wrapper to invoke.",
    )
    parser.add_argument(
        "--livecodebench-repo",
        default="",
        help="LiveCodeBench checkout used by the LiveCodeBench datasets.",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--dp", type=int, default=None)
    parser.add_argument(
        "--gpus-per-job",
        type=int,
        default=1,
        help="Number of GPUs to request from Slurm for each submitted job.",
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Fallback max-sample cap for datasets without an explicit suite cap.",
    )
    return parser.parse_args()


def _nonempty_csv(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("Expected at least one comma-separated value.")
    return values


def _selected_requires_livecodebench_repo(
    datasets: list[str],
    suite_definition,
) -> bool:
    try:
        return any(
            get_suite_dataset(
                dataset_key,
                suite_definition=suite_definition,
            ).requires_livecodebench_repo
            for dataset_key in datasets
        )
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


def _validate_livecodebench_repo(path: str) -> None:
    if not path:
        raise SystemExit(
            "Selected LiveCodeBench datasets require --livecodebench-repo pointing "
            "at the LiveCodeBench checkout root."
        )
    if not os.path.isdir(path):
        raise SystemExit(f"LiveCodeBench repo path does not exist: {path}")
    runner_dir = os.path.join(path, "lcb_runner")
    if not os.path.isdir(runner_dir):
        raise SystemExit(
            "LiveCodeBench repo path must be the checkout root containing "
            f"lcb_runner/: {path}"
        )


def _command_preview(
    env_updates: dict[str, str],
    slurm_script: str,
    job_name: str,
    *,
    gpus_per_job: int,
) -> str:
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_updates.items())
    slurm_args = [f"--job-name {shlex.quote(job_name)}"]
    if gpus_per_job != 1:
        slurm_args.append(f"--gres=gpu:{gpus_per_job}")
    return f"{assignments} sbatch {' '.join(slurm_args)} {shlex.quote(slurm_script)}"


def main() -> None:
    args = _parse_args()
    suite_definition = load_suite_definition(args.config)
    datasets = (
        _nonempty_csv(args.datasets)
        if args.datasets is not None
        else suite_dataset_keys(suite_definition)
    )
    modes = _nonempty_csv(args.thinking_modes)
    if args.gpus_per_job < 1:
        raise SystemExit("--gpus-per-job must be >= 1.")
    base_config = suite_definition.suite_config
    suite_config = replace(
        base_config,
        model_id=args.model_id if args.model_id is not None else base_config.model_id,
        temperature=(
            args.temperature if args.temperature is not None else base_config.temperature
        ),
        num_generations=(
            args.num_generations
            if args.num_generations is not None
            else base_config.num_generations
        ),
        max_tokens=args.max_tokens if args.max_tokens is not None else base_config.max_tokens,
        max_model_len=(
            args.max_model_len
            if args.max_model_len is not None
            else base_config.max_model_len
        ),
        tp=args.tp if args.tp is not None else base_config.tp,
        dp=args.dp if args.dp is not None else base_config.dp,
        dtype=args.dtype if args.dtype is not None else base_config.dtype,
        max_num_seqs=(
            args.max_num_seqs
            if args.max_num_seqs is not None
            else base_config.max_num_seqs
        ),
        max_num_batched_tokens=(
            args.max_num_batched_tokens
            if args.max_num_batched_tokens is not None
            else base_config.max_num_batched_tokens
        ),
        seed=args.seed if args.seed is not None else base_config.seed,
        max_samples=(
            args.max_samples if args.max_samples is not None else base_config.max_samples
        ),
    )
    if suite_config.dp > args.gpus_per_job:
        raise SystemExit("--dp cannot exceed --gpus-per-job.")

    if not os.path.isfile(args.slurm_script):
        raise SystemExit(f"Slurm wrapper not found: {args.slurm_script}")
    requires_livecodebench_repo = _selected_requires_livecodebench_repo(
        datasets,
        suite_definition,
    )
    if requires_livecodebench_repo or args.livecodebench_repo:
        _validate_livecodebench_repo(args.livecodebench_repo)
    os.makedirs(args.output_root, exist_ok=True)
    runtime_conda_env = (
        os.environ.get("CONDA_ENV")
        or os.environ.get("CONDA_DEFAULT_ENV")
        or ""
    )
    local_venv_activate = os.path.join(ROOT, ".venv", "bin", "activate")
    if args.submit and not runtime_conda_env and not os.path.isfile(local_venv_activate):
        raise SystemExit(
            "Submitting the suite requires a runtime env. Set CONDA_ENV, activate "
            "the desired conda env, or create .venv for the Slurm wrapper fallback."
        )
    manifest: dict[str, object] = {
        "schema_name": "main_rollout_stats_suite.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submit": bool(args.submit),
        "suite_config_path": suite_definition.config_path,
        "suite_config": asdict(suite_config),
        "jobs": [],
    }

    submitted_jobs: list[dict[str, str]] = []
    for mode in modes:
        for dataset_key in datasets:
            env_updates = build_collect_env(
                dataset_key,
                mode,
                suite_config=suite_config,
                output_root=args.output_root,
                livecodebench_repo=args.livecodebench_repo or None,
                suite_definition=suite_definition,
            )
            if args.resume:
                env_updates = dict(env_updates)
                env_updates["RESUME"] = "1"
            if runtime_conda_env:
                env_updates = dict(env_updates)
                env_updates["CONDA_ENV"] = runtime_conda_env
            job_name = f"{args.job_prefix}-{dataset_key}-{mode}"
            preview = _command_preview(
                env_updates,
                args.slurm_script,
                job_name,
                gpus_per_job=args.gpus_per_job,
            )
            manifest["jobs"].append(
                {
                    "dataset_key": dataset_key,
                    "thinking_mode": mode,
                    "job_name": job_name,
                    "gpus_per_job": args.gpus_per_job,
                    "resume": bool(args.resume),
                    "collector_env": env_updates,
                    "preview": preview,
                }
            )
            print(preview)
            if not args.submit:
                continue
            env = os.environ.copy()
            env.update(env_updates)
            sbatch_command = ["sbatch", "--job-name", job_name]
            if args.gpus_per_job != 1:
                sbatch_command.append(f"--gres=gpu:{args.gpus_per_job}")
            sbatch_command.append(args.slurm_script)
            try:
                result = subprocess.run(
                    sbatch_command,
                    cwd=ROOT,
                    check=True,
                    text=True,
                    capture_output=True,
                    env=env,
                )
            except subprocess.CalledProcessError as exc:
                if exc.stdout:
                    print(exc.stdout, end="", file=sys.stderr)
                if exc.stderr:
                    print(exc.stderr, end="", file=sys.stderr)
                raise
            submitted_jobs.append(
                {
                    "dataset_key": dataset_key,
                    "thinking_mode": mode,
                    "job_name": job_name,
                    "sbatch_stdout": result.stdout.strip(),
                }
            )
            print(result.stdout.strip())

    manifest["submitted_jobs"] = submitted_jobs
    manifest_path = os.path.join(args.output_root, args.manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote suite manifest to {manifest_path}")


if __name__ == "__main__":
    main()
