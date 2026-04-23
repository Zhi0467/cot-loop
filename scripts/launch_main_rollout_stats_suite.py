#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.main_rollout_stats_suite import (  # noqa: E402
    MainRolloutSuiteConfig,
    build_collect_env,
    suite_dataset_keys,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default=",".join(suite_dataset_keys()),
        help="Comma-separated suite dataset keys.",
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
        "--slurm-script",
        default=os.path.join(ROOT, "slurm", "run_collect_model_stats.sbatch"),
        help="Collector sbatch wrapper to invoke.",
    )
    parser.add_argument(
        "--livecodebench-repo",
        default="",
        help="LiveCodeBench checkout used by the LiveCodeBench datasets.",
    )
    parser.add_argument(
        "--lcb-extra-exclude-prompt-jsonl",
        default="",
        help="Prompt archive used to keep LiveCodeBench-extra disjoint.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=81920)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-num-seqs", type=int, default=10)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
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


def _command_preview(env_updates: dict[str, str], slurm_script: str, job_name: str) -> str:
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_updates.items())
    return f"{assignments} sbatch --job-name {shlex.quote(job_name)} {shlex.quote(slurm_script)}"


def main() -> None:
    args = _parse_args()
    datasets = _nonempty_csv(args.datasets)
    modes = _nonempty_csv(args.thinking_modes)
    suite_config = MainRolloutSuiteConfig(
        model_id=args.model_id,
        temperature=args.temperature,
        num_generations=args.num_generations,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        tp=args.tp,
        dp=args.dp,
        dtype=args.dtype,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    if not os.path.isfile(args.slurm_script):
        raise SystemExit(f"Slurm wrapper not found: {args.slurm_script}")
    if args.livecodebench_repo and not os.path.isdir(args.livecodebench_repo):
        raise SystemExit(f"LiveCodeBench repo path does not exist: {args.livecodebench_repo}")
    if args.lcb_extra_exclude_prompt_jsonl and not os.path.isfile(args.lcb_extra_exclude_prompt_jsonl):
        raise SystemExit(
            "--lcb-extra-exclude-prompt-jsonl does not exist: "
            f"{args.lcb_extra_exclude_prompt_jsonl}"
        )

    os.makedirs(args.output_root, exist_ok=True)
    runtime_conda_env = (
        os.environ.get("CONDA_ENV")
        or os.environ.get("CONDA_DEFAULT_ENV")
        or ""
    )
    if args.submit and not runtime_conda_env:
        raise SystemExit(
            "Submitting the suite requires a runtime env. Set CONDA_ENV or activate "
            "the desired conda env before running the launcher."
        )
    manifest: dict[str, object] = {
        "schema_name": "main_rollout_stats_suite.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submit": bool(args.submit),
        "suite_config": suite_config.__dict__,
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
                lcb_extra_exclude_prompt_jsonl=args.lcb_extra_exclude_prompt_jsonl or None,
            )
            if runtime_conda_env:
                env_updates = dict(env_updates)
                env_updates["CONDA_ENV"] = runtime_conda_env
            job_name = f"{args.job_prefix}-{dataset_key}-{mode}"
            preview = _command_preview(env_updates, args.slurm_script, job_name)
            manifest["jobs"].append(
                {
                    "dataset_key": dataset_key,
                    "thinking_mode": mode,
                    "job_name": job_name,
                    "collector_env": env_updates,
                    "preview": preview,
                }
            )
            print(preview)
            if not args.submit:
                continue
            env = os.environ.copy()
            env.update(env_updates)
            result = subprocess.run(
                ["sbatch", "--job-name", job_name, args.slurm_script],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )
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
    manifest_path = os.path.join(args.output_root, "suite_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote suite manifest to {manifest_path}")


if __name__ == "__main__":
    main()
