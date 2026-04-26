from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .common import (
    pca_summary_fieldnames,
    selection_summary_fieldnames,
    sharded_path,
    write_csv,
    write_json,
    write_jsonl,
)
from .model_capture import capture_vectors, load_model, resolve_device
from .pca_outputs import (
    PooledInputs,
    extend_local_outputs,
    extend_pooled_outputs,
    load_matplotlib,
)
from .selection import (
    analysis_config,
    assign_shards,
    collect_bundle_specs,
    ledger_row,
    run_selection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select strict looped rollout buckets from rollout_bundle.v1 files "
            "and, when not in selection-only mode, extract final-layer "
            "attention-write and post-attention-residual PCA trajectories "
            "across all repeated trigger n-gram occurrences."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help=(
            "A rollout_bundle.v1 path. May be a <base>.jsonl.gz bundle, its "
            "<base>.json sidecar, or a directory containing exactly one "
            "bundle. Repeat for multiple dataset/mode bundles."
        ),
    )
    parser.add_argument(
        "--bundle-root",
        action="append",
        default=[],
        help=(
            "Discover finalized bundles under this root with --bundle-glob. "
            "Root discovery requires a rollout_bundle.v1 sidecar next to each "
            "bundle and skips rank/preexisting/generated_ungraded files."
        ),
    )
    parser.add_argument("--bundle-glob", default="**/*.jsonl.gz")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--loop-n", type=int, default=30)
    parser.add_argument("--loop-k", type=int, default=20)
    parser.add_argument("--max-per-bucket", type=int, default=5)
    parser.add_argument(
        "--selection-order",
        choices=("shortest_replay", "bundle_order"),
        default="shortest_replay",
        help=(
            "Deterministic ordering used after strict bucket filtering. "
            "shortest_replay keeps the first experiment cheaper without "
            "relaxing bucket criteria."
        ),
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write selection ledgers and summaries without loading the model.",
    )
    parser.add_argument(
        "--include-final-hidden",
        action="store_true",
        help="Also capture H, the final-layer output before the model final norm.",
    )
    parser.add_argument(
        "--max-replay-tokens",
        type=int,
        default=0,
        help=(
            "Optional safety cap on prompt plus replayed completion prefix. "
            "0 means no cap. Rows over the cap are skipped before model replay "
            "but remain visible in the selection ledger."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write JSONL/CSV PCA outputs without matplotlib figures.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = collect_bundle_specs(args)
    selected_rows, summary_rows = run_selection(specs, args)

    write_json(out_dir / "analysis_config.json", analysis_config(specs=specs, args=args))
    write_jsonl(out_dir / "selection_ledger.jsonl", [ledger_row(row) for row in selected_rows])
    write_csv(out_dir / "selection_summary.csv", summary_rows, selection_summary_fieldnames())

    shard_rows, shard_loads = assign_shards(selected_rows, num_shards=args.num_shards)
    rows_for_this_shard = shard_rows[args.shard_index]
    write_json(
        out_dir / "shard_manifest.json",
        {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "total_selected_rollouts": len(selected_rows),
            "rows_in_shard": len(rows_for_this_shard),
            "shard_replay_token_loads": shard_loads,
            "shard_row_counts": [len(rows) for rows in shard_rows],
        },
    )

    if args.selection_only or not rows_for_this_shard:
        return

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = load_model(args.model_id, device=device)
    model.to(device)
    model.eval()

    plt = None if args.no_figures else load_matplotlib()
    pca_point_rows: list[dict] = []
    pca_summary_rows: list[dict] = []
    pooled_inputs: PooledInputs = defaultdict(list)
    skipped_replay_rows: list[dict] = []

    for row in rows_for_this_shard:
        if args.max_replay_tokens and row.replay_token_count > args.max_replay_tokens:
            skipped_replay_rows.append(_skipped_replay_row(row))
            continue

        capture = capture_vectors(
            model,
            row,
            device=device,
            include_final_hidden=args.include_final_hidden,
        )
        extend_local_outputs(
            row=row,
            capture=capture,
            out_dir=out_dir,
            plt=plt,
            pca_point_rows=pca_point_rows,
            pca_summary_rows=pca_summary_rows,
            pooled_inputs=pooled_inputs,
        )

    if args.num_shards == 1:
        extend_pooled_outputs(
            pooled_inputs=pooled_inputs,
            out_dir=out_dir,
            plt=plt,
            pca_point_rows=pca_point_rows,
            pca_summary_rows=pca_summary_rows,
        )

    write_jsonl(
        sharded_path(
            out_dir,
            "pca_points.jsonl",
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        ),
        pca_point_rows,
    )
    write_csv(
        sharded_path(
            out_dir,
            "pca_summary.csv",
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        ),
        pca_summary_rows,
        pca_summary_fieldnames(),
    )
    write_jsonl(out_dir / "skipped_replay_rows.jsonl", skipped_replay_rows)


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_per_bucket < 1:
        raise SystemExit("--max-per-bucket must be >= 1.")
    if args.loop_n < 1:
        raise SystemExit("--loop-n must be >= 1.")
    if args.loop_k < 2:
        raise SystemExit("--loop-k must be >= 2.")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards.")
    if "qwen3" not in args.model_id.lower() and not args.selection_only:
        raise SystemExit(
            "Full vector extraction currently supports Qwen3 checkpoints only. "
            "Use --selection-only for model-agnostic bundle selection."
        )


def _skipped_replay_row(row) -> dict:
    return {
        "selection_id": row.selection_id,
        "dataset_key": row.dataset_key,
        "thinking_mode": row.thinking_mode,
        "bucket": row.bucket,
        "sample_id": row.sample_id,
        "rollout_index": row.rollout_index,
        "replay_token_count": row.replay_token_count,
        "reason": "max_replay_tokens",
    }
