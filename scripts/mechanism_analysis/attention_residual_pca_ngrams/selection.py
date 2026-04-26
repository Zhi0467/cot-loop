from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .common import git_commit, slug
from .records import (
    BUCKET_CORRECT_IN_BUDGET,
    BUCKET_WRONG_MAX_LENGTH,
    BUCKETS,
    BundleSpec,
    SelectedRollout,
)
from probe.bundle_io import bundle_paths, iter_bundle_rows, read_bundle_sidecar


def resolve_bundle_path(path: Path) -> tuple[Path, Path | None]:
    if path.is_file():
        if path.name.endswith(".jsonl.gz"):
            sidecar, bundle = bundle_paths(path)
            sidecar_path = Path(sidecar)
            return Path(bundle), sidecar_path if sidecar_path.is_file() else None
        if path.suffix == ".json":
            sidecar, bundle = bundle_paths(path)
            bundle_path = Path(bundle)
            if not bundle_path.is_file():
                raise SystemExit(f"No bundle found next to sidecar {path}.")
            return bundle_path, Path(sidecar)
        raise SystemExit(
            f"Unsupported bundle path {path}; expected <base>.jsonl.gz or <base>.json."
        )

    if not path.is_dir():
        raise SystemExit(f"Bundle path does not exist: {path}")

    bundles = sorted(path.glob("*.jsonl.gz"))
    bundles = [bundle for bundle in bundles if is_candidate_bundle(bundle)]
    if len(bundles) == 1:
        return resolve_bundle_path(bundles[0])
    if len(bundles) > 1:
        raise SystemExit(
            f"Multiple bundles found in {path}; pass a specific <base>.jsonl.gz path."
        )
    raise SystemExit(f"No rollout bundle found in {path}.")


def is_candidate_bundle(path: Path) -> bool:
    name = path.name
    if name.endswith("__preexisting.jsonl.gz"):
        return False
    if "__rank" in name:
        return False
    if ".generated_ungraded_" in name:
        return False
    return True


def discover_bundle_paths(roots: list[str], glob_pattern: str) -> list[Path]:
    discovered: list[Path] = []
    seen: set[str] = set()
    for root_text in roots:
        root = Path(root_text)
        if not root.is_dir():
            raise SystemExit(f"--bundle-root is not a directory: {root}")
        for candidate in sorted(root.glob(glob_pattern)):
            if not candidate.is_file() or not is_candidate_bundle(candidate):
                continue
            sidecar, _bundle = bundle_paths(candidate)
            sidecar_path = Path(sidecar)
            if not sidecar_path.is_file():
                continue
            try:
                sidecar_payload = read_bundle_sidecar(str(sidecar_path))
            except (OSError, json.JSONDecodeError):
                continue
            if sidecar_payload.get("schema") != "rollout_bundle.v1":
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(candidate)
    return discovered


def bundle_spec(path: Path) -> BundleSpec:
    bundle_path, sidecar_path = resolve_bundle_path(path)
    sidecar = _load_sidecar(sidecar_path)
    metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
    dataset = str(metadata.get("dataset") or bundle_path.stem)
    thinking_mode = str(metadata.get("thinking_mode") or "unknown")
    task_kind = str(metadata.get("task_kind") or "unknown")
    model_id = metadata.get("model_id")
    loop_n, loop_k = _loop_detector_from_sidecar(sidecar)
    dataset_key = slug(
        "__".join(
            part
            for part in (
                dataset,
                str(metadata.get("release_version") or ""),
                str(metadata.get("split") or ""),
            )
            if part
        )
    )
    return BundleSpec(
        bundle_path=bundle_path,
        sidecar_path=sidecar_path,
        dataset=dataset,
        dataset_key=dataset_key,
        thinking_mode=thinking_mode,
        task_kind=task_kind,
        model_id=str(model_id) if model_id is not None else None,
        loop_n=loop_n,
        loop_k=loop_k,
    )


def collect_bundle_specs(args: Any) -> list[BundleSpec]:
    paths = [Path(path) for path in args.bundle]
    paths.extend(discover_bundle_paths(args.bundle_root, args.bundle_glob))
    if not paths:
        raise SystemExit(
            "Pass at least one --bundle or --bundle-root pointing at current "
            "rollout_bundle.v1 artifacts."
        )

    specs: list[BundleSpec] = []
    seen: set[str] = set()
    for path in paths:
        bundle_path, _sidecar_path = resolve_bundle_path(path)
        resolved = str(bundle_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        specs.append(bundle_spec(path))
    return specs


def run_selection(
    specs: list[BundleSpec],
    args: Any,
) -> tuple[list[SelectedRollout], list[dict[str, Any]]]:
    all_selected: list[SelectedRollout] = []
    all_summaries: list[dict[str, Any]] = []
    for bundle_index, spec in enumerate(specs):
        selected, summaries = select_from_bundle(
            spec,
            bundle_index=bundle_index,
            loop_n=args.loop_n,
            loop_k=args.loop_k,
            max_per_bucket=args.max_per_bucket,
            selection_order=args.selection_order,
        )
        all_selected.extend(selected)
        all_summaries.extend(summaries)
    all_selected.sort(key=lambda row: selection_sort_key(row, args.selection_order))
    all_summaries.sort(
        key=lambda row: (
            str(row["dataset_key"]),
            str(row["thinking_mode"]),
            str(row["bucket"]),
            str(row["source_bundle"]),
        )
    )
    return all_selected, all_summaries


def analysis_config(*, specs: list[BundleSpec], args: Any) -> dict[str, Any]:
    return {
        "schema": "attention_residual_pca_ngrams.v1",
        "plan_doc": "docs/weeks/2026-W17/deeper-attention-mechanism-analysis-2026-04-25.md",
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "loop_n": args.loop_n,
        "loop_k": args.loop_k,
        "max_per_bucket": args.max_per_bucket,
        "selection_order": args.selection_order,
        "selection_only": args.selection_only,
        "include_final_hidden": args.include_final_hidden,
        "max_replay_tokens": args.max_replay_tokens,
        "device": args.device,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "source_bundles": [
            {
                "bundle_path": str(spec.bundle_path),
                "sidecar_path": str(spec.sidecar_path) if spec.sidecar_path else None,
                "dataset": spec.dataset,
                "dataset_key": spec.dataset_key,
                "thinking_mode": spec.thinking_mode,
                "task_kind": spec.task_kind,
                "model_id": spec.model_id,
                "loop_n": spec.loop_n,
                "loop_k": spec.loop_k,
            }
            for spec in specs
        ],
    }


def select_from_bundle(
    spec: BundleSpec,
    *,
    bundle_index: int,
    loop_n: int,
    loop_k: int,
    max_per_bucket: int,
    selection_order: str,
) -> tuple[list[SelectedRollout], list[dict[str, Any]]]:
    eligible: dict[str, list[SelectedRollout]] = {bucket: [] for bucket in BUCKETS}
    counts = _empty_counts()
    effective_loop_n = spec.loop_n or loop_n
    effective_loop_k = spec.loop_k or loop_k

    for row in iter_bundle_rows(str(spec.bundle_path)):
        prompt_token_ids = token_list(row.get("prompt_token_ids"))
        if prompt_token_ids is None:
            prompt_token_ids = token_list(row.get("prompt_tokens"))
        split = str(row.get("split") or row.get("source_split") or "unknown")
        sample_id = str(row.get("sample_id") if row.get("sample_id") is not None else "")
        record_id = str(row.get("record_id") or sample_id)
        rollouts = row.get("rollouts")
        if not isinstance(rollouts, list):
            continue

        for fallback_rollout_index, rollout in enumerate(rollouts):
            if not isinstance(rollout, dict):
                continue
            selected = _maybe_select_rollout(
                rollout=rollout,
                fallback_rollout_index=fallback_rollout_index,
                counts=counts,
                spec=spec,
                bundle_index=bundle_index,
                prompt_token_ids=prompt_token_ids,
                split=split,
                sample_id=sample_id,
                record_id=record_id,
                default_loop_n=effective_loop_n,
                default_loop_k=effective_loop_k,
            )
            if selected is not None:
                eligible[selected.bucket].append(selected)

    selected_rows: list[SelectedRollout] = []
    summary_rows: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        rows = sorted(
            eligible[bucket],
            key=lambda row: selection_sort_key(row, selection_order),
        )
        selected_bucket_rows = rows[:max_per_bucket]
        selected_rows.extend(selected_bucket_rows)
        summary_rows.append(
            {
                "dataset": spec.dataset,
                "dataset_key": spec.dataset_key,
                "thinking_mode": spec.thinking_mode,
                "task_kind": spec.task_kind,
                "source_bundle": str(spec.bundle_path),
                "bucket": bucket,
                "eligible_rollouts": len(rows),
                "selected_rollouts": len(selected_bucket_rows),
                "requested_rollouts": max_per_bucket,
                "underfilled": len(selected_bucket_rows) < max_per_bucket,
                **counts,
            }
        )
    selected_rows.sort(key=lambda row: selection_sort_key(row, selection_order))
    return selected_rows, summary_rows


def ledger_row(row: SelectedRollout) -> dict[str, Any]:
    return {
        "selection_id": row.selection_id,
        "dataset": row.dataset,
        "dataset_key": row.dataset_key,
        "thinking_mode": row.thinking_mode,
        "task_kind": row.task_kind,
        "source_bundle": row.source_bundle,
        "source_sidecar": row.source_sidecar,
        "split": row.split,
        "sample_id": row.sample_id,
        "record_id": row.record_id,
        "rollout_index": row.rollout_index,
        "bucket": row.bucket,
        "correct": row.correct,
        "max_length_hit": row.max_length_hit,
        "finish_reason": row.finish_reason,
        "prompt_token_count": row.prompt_token_count,
        "completion_token_count": row.completion_token_count,
        "replay_token_count": row.replay_token_count,
        "loop_n": row.loop_n,
        "loop_k": row.loop_k,
        "ngram_length": len(row.ngram_token_ids),
        "ngram_token_ids": row.ngram_token_ids,
        "saved_ngram_start_positions": row.saved_ngram_start_positions,
        "rescan_ngram_start_positions": row.rescan_ngram_start_positions,
        "num_ngram_occurrences": len(row.rescan_ngram_start_positions),
        "boundary_positions": row.boundary_positions,
        "first_loop_prefix_length": row.first_loop_prefix_length,
        "grading": row.grading,
    }


def assign_shards(
    rows: list[SelectedRollout],
    *,
    num_shards: int,
) -> tuple[list[list[SelectedRollout]], list[int]]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.replay_token_count,
            row.completion_token_count,
            row.dataset_key,
            row.thinking_mode,
            row.sample_id,
            row.rollout_index,
        ),
        reverse=True,
    )
    shard_rows: list[list[SelectedRollout]] = [[] for _ in range(num_shards)]
    shard_loads = [0 for _ in range(num_shards)]
    for row in ordered_rows:
        target = min(
            range(num_shards),
            key=lambda shard_idx: (shard_loads[shard_idx], len(shard_rows[shard_idx])),
        )
        shard_rows[target].append(row)
        shard_loads[target] += row.replay_token_count
    for rows_for_shard in shard_rows:
        rows_for_shard.sort(
            key=lambda row: (
                row.dataset_key,
                row.thinking_mode,
                row.bucket,
                row.replay_token_count,
                row.sample_id,
                row.rollout_index,
            )
        )
    return shard_rows, shard_loads


def find_all_ngram_starts(token_ids: list[int], ngram: list[int]) -> list[int]:
    n = len(ngram)
    if n == 0 or len(token_ids) < n:
        return []
    starts: list[int] = []
    first = ngram[0]
    for start in range(0, len(token_ids) - n + 1):
        if token_ids[start] != first:
            continue
        if token_ids[start : start + n] == ngram:
            starts.append(start)
    return starts


def selection_sort_key(row: SelectedRollout, order: str) -> tuple[Any, ...]:
    base = (
        row.dataset_key,
        row.thinking_mode,
        row.sample_id,
        row.rollout_index,
        row.selection_id,
    )
    if order == "bundle_order":
        return base
    return (row.replay_token_count, row.completion_token_count, *base)


def token_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        if not isinstance(item, int):
            return None
        out.append(int(item))
    return out


def _maybe_select_rollout(
    *,
    rollout: dict[str, Any],
    fallback_rollout_index: int,
    counts: dict[str, int],
    spec: BundleSpec,
    bundle_index: int,
    prompt_token_ids: list[int] | None,
    split: str,
    sample_id: str,
    record_id: str,
    default_loop_n: int,
    default_loop_k: int,
) -> SelectedRollout | None:
    counts["total_rollouts"] += 1
    if not _parse_boolish(rollout.get("loop_flag")):
        return None
    counts["loop_rollouts"] += 1
    if prompt_token_ids is None:
        counts["missing_prompt_token_ids"] += 1
        return None
    completion_token_ids = token_list(rollout.get("completion_token_ids"))
    if completion_token_ids is None:
        counts["missing_completion_token_ids"] += 1
        return None
    correct = _rollout_correct(rollout, spec.task_kind)
    if correct is None:
        counts["missing_grading"] += 1
        return None

    trigger_parts = _loop_trigger_parts(
        rollout,
        default_n=default_loop_n,
        default_k=default_loop_k,
    )
    if trigger_parts is None:
        counts["missing_loop_trigger"] += 1
        return None
    trigger_loop_n, trigger_loop_k, ngram, saved_starts, first_loop_prefix = (
        trigger_parts
    )
    if len(ngram) != trigger_loop_n:
        counts["invalid_loop_trigger"] += 1
        return None
    rescan_starts = find_all_ngram_starts(completion_token_ids, ngram)
    if [start for start in saved_starts if start not in rescan_starts]:
        counts["ngram_rescan_miss"] += 1
        return None
    if not rescan_starts:
        counts["invalid_loop_trigger"] += 1
        return None

    prompt_len = len(prompt_token_ids)
    boundary_positions = [prompt_len + start - 1 for start in rescan_starts]
    if prompt_len < 1 or any(boundary < 0 for boundary in boundary_positions):
        counts["boundary_mapping_invalid"] += 1
        return None
    max_length_hit = _rollout_max_length_hit(rollout)
    if correct and not max_length_hit:
        bucket = BUCKET_CORRECT_IN_BUDGET
    elif (not correct) and max_length_hit:
        bucket = BUCKET_WRONG_MAX_LENGTH
    else:
        counts["other_loop_rollouts"] += 1
        return None

    rollout_index = int(
        rollout.get(
            "rollout_index",
            rollout.get("generation_index", fallback_rollout_index),
        )
    )
    return SelectedRollout(
        selection_id=_selection_id(
            bundle_path=spec.bundle_path,
            sample_id=sample_id,
            rollout_index=rollout_index,
            bucket=bucket,
        ),
        bundle_index=bundle_index,
        source_bundle=str(spec.bundle_path),
        source_sidecar=str(spec.sidecar_path) if spec.sidecar_path is not None else None,
        dataset=spec.dataset,
        dataset_key=spec.dataset_key,
        thinking_mode=spec.thinking_mode,
        task_kind=spec.task_kind,
        split=split,
        sample_id=sample_id,
        record_id=record_id,
        rollout_index=rollout_index,
        bucket=bucket,
        correct=bool(correct),
        max_length_hit=bool(max_length_hit),
        finish_reason=str(rollout.get("finish_reason") or "unknown"),
        prompt_token_ids=[int(token_id) for token_id in prompt_token_ids],
        completion_token_ids=[int(token_id) for token_id in completion_token_ids],
        prompt_token_count=prompt_len,
        completion_token_count=len(completion_token_ids),
        replay_token_count=prompt_len + max(rescan_starts),
        loop_n=trigger_loop_n,
        loop_k=trigger_loop_k,
        ngram_token_ids=ngram,
        saved_ngram_start_positions=saved_starts,
        rescan_ngram_start_positions=rescan_starts,
        boundary_positions=boundary_positions,
        first_loop_prefix_length=first_loop_prefix,
        grading=dict(rollout.get("grading") or {}),
    )


def _empty_counts() -> dict[str, int]:
    return {
        "total_rollouts": 0,
        "loop_rollouts": 0,
        "other_loop_rollouts": 0,
        "missing_grading": 0,
        "missing_completion_token_ids": 0,
        "missing_prompt_token_ids": 0,
        "missing_loop_trigger": 0,
        "invalid_loop_trigger": 0,
        "ngram_rescan_miss": 0,
        "boundary_mapping_invalid": 0,
    }


def _load_sidecar(sidecar_path: Path | None) -> dict[str, Any]:
    if sidecar_path is None:
        return {}
    try:
        return read_bundle_sidecar(str(sidecar_path))
    except (OSError, json.JSONDecodeError):
        return {}


def _loop_detector_from_sidecar(sidecar: dict[str, Any]) -> tuple[int | None, int | None]:
    detector = sidecar.get("metadata", {}).get("loop_detector")
    if not isinstance(detector, dict):
        return None, None
    loop_n = detector.get("n")
    loop_k = detector.get("k")
    return (
        int(loop_n) if isinstance(loop_n, int) else None,
        int(loop_k) if isinstance(loop_k, int) else None,
    )


def _parse_boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "passed", "correct"}:
            return True
        if normalized in {"false", "0", "no", "failed", "incorrect"}:
            return False
    return None


def _rollout_correct(rollout: dict[str, Any], task_kind: str) -> bool | None:
    grading = rollout.get("grading")
    if not isinstance(grading, dict):
        return None
    if task_kind in {"livecodebench_codegen", "taco_codegen"}:
        return _parse_boolish(grading.get("passed"))
    if "correct" in grading:
        return _parse_boolish(grading.get("correct"))
    if "passed" in grading:
        return _parse_boolish(grading.get("passed"))
    return None


def _rollout_max_length_hit(rollout: dict[str, Any]) -> bool:
    value = _parse_boolish(rollout.get("max_length_hit"))
    if value is not None:
        return value
    finish_reason = str(rollout.get("finish_reason") or "").strip().lower()
    if finish_reason:
        return finish_reason == "length"
    return bool(int(rollout.get("cap_hit") or 0))


def _loop_trigger_parts(
    rollout: dict[str, Any],
    *,
    default_n: int,
    default_k: int,
) -> tuple[int, int, list[int], list[int], int | None] | None:
    trigger = rollout.get("loop_trigger")
    if not isinstance(trigger, dict):
        return None

    ngram = token_list(trigger.get("ngram_token_ids"))
    if ngram is None:
        ngram = token_list(trigger.get("ngram"))
    if not ngram:
        return None

    starts = token_list(trigger.get("ngram_start_positions"))
    if starts is None:
        starts = token_list(trigger.get("start_positions"))
    if starts is None:
        starts = []
    if not starts and isinstance(trigger.get("trigger_start"), int):
        starts = [int(trigger["trigger_start"])]

    loop_n = int(trigger.get("n") or default_n or len(ngram))
    loop_k = int(trigger.get("k") or default_k)
    first_loop_prefix = rollout.get("first_loop_prefix_length")
    if first_loop_prefix is None and isinstance(trigger.get("trigger_end"), int):
        first_loop_prefix = int(trigger["trigger_end"]) + 1
    return (
        loop_n,
        loop_k,
        [int(token_id) for token_id in ngram],
        [int(start) for start in starts],
        int(first_loop_prefix) if first_loop_prefix is not None else None,
    )


def _selection_id(
    *,
    bundle_path: Path,
    sample_id: str,
    rollout_index: int,
    bucket: str,
) -> str:
    digest = hashlib.sha1(
        f"{bundle_path}:{sample_id}:{rollout_index}:{bucket}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{slug(bucket)}__s{slug(sample_id)}__r{rollout_index}__{digest}"
