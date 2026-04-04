#!/usr/bin/env python3
"""Run the locked prompt-profile full-train surface for the CoT loop project."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "full_train"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    task_kind: str
    dataset: str
    prompt_field: str
    train_split: str
    test_split: str
    train_max_samples: int
    test_max_samples: int
    train_config: str | None = None
    test_config: str | None = None
    tp: int = 1
    dp: int = 1
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 1024
    release_version: str | None = None
    needs_livecodebench_repo: bool = False


DATASET_SPECS: dict[str, DatasetSpec] = {
    "gpqa": DatasetSpec(
        key="gpqa",
        display_name="GPQA",
        task_kind="multiple_choice_gpqa",
        dataset="data/gpqa_diamond.csv",
        prompt_field="Question",
        train_split="train",
        test_split="train",
        train_max_samples=158,
        test_max_samples=40,
        train_config="gpqa_diamond",
        test_config="gpqa_diamond",
    ),
    "aime": DatasetSpec(
        key="aime",
        display_name="AIME",
        task_kind="math_freeform",
        dataset="data/aime_2024_2025.jsonl",
        prompt_field="question",
        train_split="test",
        test_split="test",
        train_max_samples=48,
        test_max_samples=12,
    ),
    "math500": DatasetSpec(
        key="math500",
        display_name="MATH-500",
        task_kind="math_freeform",
        dataset="HuggingFaceH4/MATH-500",
        prompt_field="problem",
        train_split="test",
        test_split="test",
        train_max_samples=400,
        test_max_samples=100,
    ),
    "mmlu_pro": DatasetSpec(
        key="mmlu_pro",
        display_name="MMLU-Pro",
        task_kind="multiple_choice_mmlupro",
        dataset="TIGER-Lab/MMLU-Pro",
        prompt_field="problem",
        train_split="test",
        test_split="test",
        train_max_samples=640,
        test_max_samples=160,
    ),
    "livecodebench": DatasetSpec(
        key="livecodebench",
        display_name="LiveCodeBench",
        task_kind="livecodebench_codegen",
        dataset="livecodebench_release_v6",
        prompt_field="problem",
        train_split="test",
        test_split="test",
        train_max_samples=640,
        test_max_samples=160,
        dp=2,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        release_version="release_v6",
        needs_livecodebench_repo=True,
    ),
}

REGRESSION_VIEW_ARGS: dict[str, list[str]] = {
    "ensemble": ["--classifier-mode", "ensemble", "--score-rule", "mean_prob"],
    "last_layer": ["--classifier-mode", "last_layer", "--classifier-layer", "-1"],
}

BINARY_VIEW_ARGS: dict[str, list[str]] = {
    "ensemble": ["--classifier-mode", "ensemble", "--score-rule", "vote_fraction"],
    "last_layer": ["--classifier-mode", "last_layer", "--classifier-layer", "-1"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help="Datasets to run: gpqa, aime, math500, mmlu_pro, livecodebench, or all.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "build", "regression", "relabel", "binary", "summary"),
        default="all",
        help="Which portion of the full-train surface to execute.",
    )
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Defaults to <out-root>/summary when omitted.",
    )
    parser.add_argument(
        "--source-out-root",
        default=None,
        help=(
            "Optional root to read existing shared archives from while writing "
            "new runs under --out-root."
        ),
    )
    parser.add_argument(
        "--regression-data-mode",
        choices=("natural", "reuse_binary_subset"),
        default="natural",
        help=(
            "Use the natural shared-archive regression data, or relabel "
            "mean_relative_length onto the balanced binary prompt subset."
        ),
    )
    parser.add_argument(
        "--regression-subset-root",
        default=None,
        help=(
            "When --regression-data-mode=reuse_binary_subset, read sample IDs from "
            "<root>/<dataset>/majority_s_0.5/data. Defaults to --out-root."
        ),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--wandb-project", default="cot-loop-probe")
    parser.add_argument("--probe-preset", default="mlp")
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=None,
        help="Optional hidden width override when --probe-preset=mlp.",
    )
    parser.add_argument(
        "--mlp-depth",
        type=int,
        default=None,
        help="Optional hidden-layer count override when --probe-preset=mlp.",
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=None,
        help="Optional dropout override when --probe-preset=mlp.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--loop-n", type=int, default=30)
    parser.add_argument("--loop-k", type=int, default=20)
    parser.add_argument("--feature-pooling", default="last_token_all_layers_stack")
    parser.add_argument("--feature-layer", type=int, default=-1)
    parser.add_argument(
        "--livecodebench-repo",
        default="",
        help="Required when running the LiveCodeBench dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def resolve_datasets(requested: list[str]) -> list[DatasetSpec]:
    if not requested or requested == ["all"]:
        return [DATASET_SPECS[key] for key in DATASET_SPECS]
    resolved: list[DatasetSpec] = []
    for item in requested:
        key = item.strip().lower()
        if key == "all":
            return [DATASET_SPECS[name] for name in DATASET_SPECS]
        if key not in DATASET_SPECS:
            valid = ", ".join(DATASET_SPECS)
            raise SystemExit(f"Unknown --dataset '{item}'. Valid: {valid}, all")
        resolved.append(DATASET_SPECS[key])
    return resolved


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(f"$ {format_command(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_root(out_root: Path, spec: DatasetSpec) -> Path:
    return out_root / spec.key


def shared_archive_dir(out_root: Path, spec: DatasetSpec) -> Path:
    return dataset_root(out_root, spec) / "shared_archive"


def source_shared_archive_dir(
    args: argparse.Namespace,
    out_root: Path,
    spec: DatasetSpec,
) -> Path:
    source_root = (
        Path(args.source_out_root).resolve()
        if args.source_out_root
        else out_root.resolve()
    )
    return shared_archive_dir(source_root, spec)


def regression_run_dir(out_root: Path, spec: DatasetSpec, view_name: str) -> Path:
    return dataset_root(out_root, spec) / "mean_relative_length" / view_name


def regression_data_dir(out_root: Path, spec: DatasetSpec) -> Path:
    return dataset_root(out_root, spec) / "mean_relative_length" / "data"


def binary_data_dir(out_root: Path, spec: DatasetSpec) -> Path:
    return dataset_root(out_root, spec) / "majority_s_0.5" / "data"


def binary_run_dir(out_root: Path, spec: DatasetSpec, view_name: str) -> Path:
    return dataset_root(out_root, spec) / "majority_s_0.5" / view_name


def manifest_path(data_dir: Path) -> Path:
    return data_dir / "manifest.json"


def relabel_target_spec(source_manifest: dict[str, Any]) -> dict[str, Any]:
    rollout_cfg = source_manifest.get("rollout_config")
    if not isinstance(rollout_cfg, dict):
        raise SystemExit("Source manifest is missing rollout_config for relabel comparison.")
    num_generations = rollout_cfg.get("num_generations")
    if not isinstance(num_generations, int):
        raise SystemExit("Source manifest rollout_config is missing integer num_generations.")
    return {
        "kind": "binary",
        "source": "prompt_profile",
        "name": "majority_s_0.5",
        "profile_target": "majority_tail",
        "tail_threshold": 0.5,
        "num_generations": num_generations,
        "positive_rule": "strict_majority",
    }


def regression_target_spec(source_manifest: dict[str, Any]) -> dict[str, Any]:
    rollout_cfg = source_manifest.get("rollout_config")
    if not isinstance(rollout_cfg, dict):
        raise SystemExit("Source manifest is missing rollout_config for regression relabel.")
    num_generations = rollout_cfg.get("num_generations")
    if not isinstance(num_generations, int):
        raise SystemExit("Source manifest rollout_config is missing integer num_generations.")
    return {
        "kind": "regression",
        "name": "mean_relative_length",
        "profile_target": "mean_relative_length",
        "tail_threshold": 0.5,
        "num_generations": num_generations,
        "loss": "sigmoid_mse",
    }


RELABEL_SOURCE_MANIFEST_KEYS = (
    "version",
    "input_dim",
    "sample_shape",
    "default_feature_key",
    "feature_extraction",
    "feature_views",
    "task_kind",
    "prompt_field",
    "answer_field",
    "prompt_template",
    "split_source",
    "split_ratio",
    "seed",
    "loop_detector",
    "rollout_config",
    "train_spec",
    "test_spec",
    "task_loader_config",
)


def relabel_output_matches_source(
    *,
    source_dir: Path,
    out_dir: Path,
    dataset_seed: int,
) -> bool:
    out_manifest_path = manifest_path(out_dir)
    source_manifest_path = manifest_path(source_dir)
    if not out_manifest_path.exists() or not source_manifest_path.exists():
        return False

    source_manifest = read_json(source_manifest_path)
    out_manifest = read_json(out_manifest_path)
    expected_archive_file = source_manifest.get("prompt_rollout_archive_file")
    if not isinstance(expected_archive_file, str) or not expected_archive_file:
        return False

    expected_values: dict[str, Any] = {
        "target_spec": relabel_target_spec(source_manifest),
        "balancing": {
            "train": "downsample",
            "test": "none",
            "seed": dataset_seed,
        },
        "prompt_profile_source_dir": str(source_dir),
        "prompt_profile_source_archive_file": expected_archive_file,
    }
    for key in RELABEL_SOURCE_MANIFEST_KEYS:
        expected_values[key] = source_manifest.get(key)

    for key, expected in expected_values.items():
        if out_manifest.get(key) != expected:
            return False
    return True


def regression_subset_source_dir(
    args: argparse.Namespace,
    *,
    out_root: Path,
    spec: DatasetSpec,
) -> Path | None:
    if args.regression_data_mode != "reuse_binary_subset":
        return None
    root = (
        Path(args.regression_subset_root).resolve()
        if args.regression_subset_root
        else out_root.resolve()
    )
    return binary_data_dir(root, spec)


def regression_data_matches_source(
    *,
    source_dir: Path,
    out_dir: Path,
    subset_source_dir: Path,
) -> bool:
    out_manifest_path = manifest_path(out_dir)
    source_manifest_path = manifest_path(source_dir)
    subset_manifest_path = manifest_path(subset_source_dir)
    if (
        not out_manifest_path.exists()
        or not source_manifest_path.exists()
        or not subset_manifest_path.exists()
    ):
        return False

    source_manifest = read_json(source_manifest_path)
    out_manifest = read_json(out_manifest_path)
    expected_archive_file = source_manifest.get("prompt_rollout_archive_file")
    if not isinstance(expected_archive_file, str) or not expected_archive_file:
        return False

    expected_values: dict[str, Any] = {
        "target_spec": regression_target_spec(source_manifest),
        "balancing": {
            "train": "none",
            "test": "none",
            "seed": 0,
        },
        "prompt_profile_source_dir": str(source_dir),
        "prompt_profile_source_archive_file": expected_archive_file,
        "sample_id_subset_source": {
            "train": str(subset_source_dir),
            "test": str(subset_source_dir),
        },
    }
    for key in RELABEL_SOURCE_MANIFEST_KEYS:
        expected_values[key] = source_manifest.get(key)

    for key, expected in expected_values.items():
        if out_manifest.get(key) != expected:
            return False
    return True


def train_run_complete(run_dir: Path, seeds: list[int]) -> bool:
    if len(seeds) == 1:
        return (
            (run_dir / "best_loss_metrics.json").exists()
            and (run_dir / "best_rank_metrics.json").exists()
        )
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        if not (
            (seed_dir / "best_loss_metrics.json").exists()
            and (seed_dir / "best_rank_metrics.json").exists()
        ):
            return False
    return True


def ensure_not_partial(run_dir: Path, seeds: list[int]) -> None:
    if not run_dir.exists():
        return
    if train_run_complete(run_dir, seeds):
        return
    raise SystemExit(
        f"Refusing to reuse partial run directory: {run_dir}. "
        "Remove it or choose a fresh --out-root."
    )


def build_dataset_cmd(
    args: argparse.Namespace,
    spec: DatasetSpec,
    *,
    out_dir: Path,
) -> list[str]:
    cmd = [
        args.python_bin,
        str(SCRIPTS_DIR / "build_probe_dataset.py"),
        "--train-dataset",
        spec.dataset,
        "--test-dataset",
        spec.dataset,
        "--train-split",
        spec.train_split,
        "--test-split",
        spec.test_split,
        "--prompt-field",
        spec.prompt_field,
        "--task-kind",
        spec.task_kind,
        "--train-max-samples",
        str(spec.train_max_samples),
        "--test-max-samples",
        str(spec.test_max_samples),
        "--seed",
        str(args.dataset_seed),
        "--model-id",
        args.model_id,
        "--temperature",
        str(args.temperature),
        "--num-generations",
        str(args.num_generations),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--tp",
        str(spec.tp),
        "--dp",
        str(spec.dp),
        "--max-num-seqs",
        str(spec.max_num_seqs),
        "--max-num-batched-tokens",
        str(spec.max_num_batched_tokens),
        "--loop-n",
        str(args.loop_n),
        "--loop-k",
        str(args.loop_k),
        "--target-kind",
        "regression",
        "--profile-target",
        "mean_relative_length",
        "--feature-pooling",
        args.feature_pooling,
        "--feature-layer",
        str(args.feature_layer),
        "--out-dir",
        str(out_dir),
        "--reuse-if-compatible",
    ]
    if spec.train_config:
        cmd.extend(["--train-config", spec.train_config])
    if spec.test_config:
        cmd.extend(["--test-config", spec.test_config])
    if spec.needs_livecodebench_repo:
        repo = args.livecodebench_repo.strip()
        if not repo:
            raise SystemExit(
                "LiveCodeBench is selected but --livecodebench-repo was not provided."
            )
        cmd.extend(["--livecodebench-repo", repo])
        if spec.release_version:
            cmd.extend(["--release-version", spec.release_version])
    return cmd


def train_probe_cmd(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    out_dir: Path,
    seed: int,
    target_kind: str,
    view_name: str,
    wandb_run_name: str,
) -> list[str]:
    view_args = BINARY_VIEW_ARGS if target_kind == "binary" else REGRESSION_VIEW_ARGS
    cmd = [
        args.python_bin,
        str(SCRIPTS_DIR / "train_probe.py"),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--probe-preset",
        args.probe_preset,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--eval-every",
        "1",
        "--log-every",
        "5",
        "--wandb-project",
        args.wandb_project,
        "--wandb-run-name",
        wandb_run_name,
    ]
    if args.mlp_hidden_dim is not None:
        cmd.extend(["--mlp-hidden-dim", str(args.mlp_hidden_dim)])
    if args.mlp_depth is not None:
        cmd.extend(["--mlp-depth", str(args.mlp_depth)])
    if args.mlp_dropout is not None:
        cmd.extend(["--mlp-dropout", str(args.mlp_dropout)])
    cmd.extend(view_args[view_name])
    return cmd


def relabel_cmd(
    args: argparse.Namespace,
    *,
    source_dir: Path,
    out_dir: Path,
) -> list[str]:
    return [
        args.python_bin,
        str(SCRIPTS_DIR / "relabel_prompt_profile_dataset.py"),
        "--source-dir",
        str(source_dir),
        "--out-dir",
        str(out_dir),
        "--target-kind",
        "binary",
        "--profile-target",
        "majority_tail",
        "--profile-tail-threshold",
        "0.5",
        "--balance-train",
        "downsample",
        "--balance-test",
        "none",
        "--balance-seed",
        str(args.dataset_seed),
    ]


def regression_relabel_cmd(
    args: argparse.Namespace,
    *,
    source_dir: Path,
    out_dir: Path,
    subset_source_dir: Path,
) -> list[str]:
    return [
        args.python_bin,
        str(SCRIPTS_DIR / "relabel_prompt_profile_dataset.py"),
        "--source-dir",
        str(source_dir),
        "--out-dir",
        str(out_dir),
        "--target-kind",
        "regression",
        "--profile-target",
        "mean_relative_length",
        "--reuse-train-sample-ids-from",
        str(subset_source_dir),
        "--reuse-test-sample-ids-from",
        str(subset_source_dir),
    ]


def summary_cmd(
    args: argparse.Namespace,
    *,
    out_root: Path,
    summary_dir: Path,
    datasets: list[DatasetSpec],
) -> list[str]:
    cmd = [
        args.python_bin,
        str(SCRIPTS_DIR / "summarize_prompt_profile_full_train.py"),
        "--out-root",
        str(out_root),
        "--summary-dir",
        str(summary_dir),
    ]
    for spec in datasets:
        cmd.extend(["--dataset", spec.key])
    return cmd


def run_build(args: argparse.Namespace, spec: DatasetSpec, out_root: Path) -> None:
    out_dir = shared_archive_dir(out_root, spec)
    cmd = build_dataset_cmd(args, spec, out_dir=out_dir)
    run_command(cmd, cwd=ROOT, dry_run=args.dry_run)


def run_regression(args: argparse.Namespace, spec: DatasetSpec, out_root: Path) -> None:
    source_dir = source_shared_archive_dir(args, out_root, spec)
    data_dir = source_dir
    if args.regression_data_mode == "reuse_binary_subset":
        subset_source_dir = regression_subset_source_dir(args, out_root=out_root, spec=spec)
        if subset_source_dir is None or not manifest_path(subset_source_dir).exists():
            raise SystemExit(
                f"Missing balanced binary sample-id source for {spec.display_name}: "
                f"{subset_source_dir}"
            )
        data_dir = regression_data_dir(out_root, spec)
        if regression_data_matches_source(
            source_dir=source_dir,
            out_dir=data_dir,
            subset_source_dir=subset_source_dir,
        ):
            print(f"[skip] regression data {spec.key} already matches balanced subset", flush=True)
        else:
            if data_dir.exists():
                print(f"[rebuild] regression data {spec.key} from balanced binary subset", flush=True)
                if not args.dry_run:
                    shutil.rmtree(data_dir)
            cmd = regression_relabel_cmd(
                args,
                source_dir=source_dir,
                out_dir=data_dir,
                subset_source_dir=subset_source_dir,
            )
            run_command(cmd, cwd=ROOT, dry_run=args.dry_run)
    if not manifest_path(data_dir).exists() and not args.dry_run:
        raise SystemExit(
            f"Missing regression data manifest for {spec.display_name}: {data_dir}"
        )
    for view_name in ("ensemble", "last_layer"):
        root_run_dir = regression_run_dir(out_root, spec, view_name)
        if train_run_complete(root_run_dir, args.seeds):
            print(f"[skip] regression {spec.key}/{view_name} already complete", flush=True)
            continue
        ensure_not_partial(root_run_dir, args.seeds)
        if len(args.seeds) == 1:
            cmd = train_probe_cmd(
                args,
                data_dir=data_dir,
                out_dir=root_run_dir,
                seed=args.seeds[0],
                target_kind="regression",
                view_name=view_name,
                wandb_run_name=f"{spec.key}-mean_relative_length-{view_name}-seed{args.seeds[0]}",
            )
            run_command(cmd, cwd=ROOT, dry_run=args.dry_run)
            continue
        for seed in args.seeds:
            seed_run_dir = root_run_dir / f"seed_{seed}"
            cmd = train_probe_cmd(
                args,
                data_dir=data_dir,
                out_dir=seed_run_dir,
                seed=seed,
                target_kind="regression",
                view_name=view_name,
                wandb_run_name=f"{spec.key}-mean_relative_length-{view_name}-seed{seed}",
            )
            run_command(cmd, cwd=ROOT, dry_run=args.dry_run)


def run_relabel(args: argparse.Namespace, spec: DatasetSpec, out_root: Path) -> None:
    source_dir = source_shared_archive_dir(args, out_root, spec)
    out_dir = binary_data_dir(out_root, spec)
    if relabel_output_matches_source(
        source_dir=source_dir,
        out_dir=out_dir,
        dataset_seed=args.dataset_seed,
    ):
        print(f"[skip] relabel {spec.key} already complete", flush=True)
        return
    if out_dir.exists():
        print(f"[rebuild] relabel {spec.key} from refreshed shared archive", flush=True)
        if not args.dry_run:
            shutil.rmtree(out_dir)
    cmd = relabel_cmd(args, source_dir=source_dir, out_dir=out_dir)
    run_command(cmd, cwd=ROOT, dry_run=args.dry_run)


def run_binary(args: argparse.Namespace, spec: DatasetSpec, out_root: Path) -> None:
    data_dir = binary_data_dir(out_root, spec)
    if not manifest_path(data_dir).exists() and not args.dry_run:
        raise SystemExit(f"Missing relabeled binary data dir for {spec.display_name}: {data_dir}")
    for view_name in ("ensemble", "last_layer"):
        root_run_dir = binary_run_dir(out_root, spec, view_name)
        if train_run_complete(root_run_dir, args.seeds):
            print(f"[skip] binary {spec.key}/{view_name} already complete", flush=True)
            continue
        ensure_not_partial(root_run_dir, args.seeds)
        if len(args.seeds) == 1:
            cmd = train_probe_cmd(
                args,
                data_dir=data_dir,
                out_dir=root_run_dir,
                seed=args.seeds[0],
                target_kind="binary",
                view_name=view_name,
                wandb_run_name=f"{spec.key}-majority_s_0.5-{view_name}-seed{args.seeds[0]}",
            )
            run_command(cmd, cwd=ROOT, dry_run=args.dry_run)
            continue
        for seed in args.seeds:
            seed_run_dir = root_run_dir / f"seed_{seed}"
            cmd = train_probe_cmd(
                args,
                data_dir=data_dir,
                out_dir=seed_run_dir,
                seed=seed,
                target_kind="binary",
                view_name=view_name,
                wandb_run_name=f"{spec.key}-majority_s_0.5-{view_name}-seed{seed}",
            )
            run_command(cmd, cwd=ROOT, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    selected_datasets = resolve_datasets(args.dataset)
    out_root = Path(args.out_root).resolve()
    summary_dir = (
        Path(args.summary_dir).resolve()
        if args.summary_dir
        else (out_root / "summary").resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "build"}:
        for spec in selected_datasets:
            run_build(args, spec, out_root)
    if args.stage in {"all", "regression"}:
        for spec in selected_datasets:
            run_regression(args, spec, out_root)
    if args.stage in {"all", "relabel"}:
        for spec in selected_datasets:
            run_relabel(args, spec, out_root)
    if args.stage in {"all", "binary"}:
        for spec in selected_datasets:
            run_binary(args, spec, out_root)
    if args.stage in {"all", "summary"}:
        cmd = summary_cmd(
            args,
            out_root=out_root,
            summary_dir=summary_dir,
            datasets=selected_datasets,
        )
        run_command(cmd, cwd=ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
