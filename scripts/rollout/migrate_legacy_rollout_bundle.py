#!/usr/bin/env python3
"""Emit a ``rollout_bundle.v1`` pair from legacy sibling artifacts.

Given an input directory that contains one or more ``<base>.json`` aggregate
stats files plus their sidecar siblings (``__lcb_records.json``,
``__prompt_rollout_archive.jsonl(.gz)``, ``__rollout_archive.jsonl.gz``,
``__prompt_profile.jsonl``), produce a single ``<base>.jsonl.gz`` bundle and a
refreshed ``<base>.json`` sidecar in ``--out-dir``.

The adapter is lossy by design:

* If a raw ``completion_text`` is unavailable (e.g. the March v2 LCB bundle
  never persisted raw rollout text), rows carry ``completion_text = null``,
  ``completion_token_ids = null``, and a row-level ``degraded`` marker so
  downstream code can detect reduced-fidelity bundles.
* ``prompt`` and ``prompt_token_ids`` may also be ``null`` when the sibling
  archives did not record them.

The script is read-only against the input directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from probe.bundle_io import (  # noqa: E402
    BUNDLE_SCHEMA,
    BundleWriter,
    bundle_paths,
    write_bundle_sidecar,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy rollout-stats artifacts to rollout_bundle.v1.",
    )
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <base>.jsonl.gz / <base>.json in --out-dir.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Optional comma-separated list of base names to restrict migration to "
            "(without the .json suffix). Defaults to every stats JSON found."
        ),
    )
    return parser.parse_args()


def _iter_legacy_bases(in_dir: Path) -> list[Path]:
    bases: list[Path] = []
    for path in sorted(in_dir.glob("*.json")):
        if path.name.endswith("__lcb_records.json"):
            continue
        if path.name.endswith("__progress.json"):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
            bases.append(path)
    return bases


def _open_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def _resolve_sibling(base: Path, suffix: str) -> Path | None:
    stem = base.stem
    candidate = base.with_name(f"{stem}{suffix}")
    return candidate if candidate.is_file() else None


def _load_lcb_records(base: Path) -> list[dict[str, Any]] | None:
    sidecar = _resolve_sibling(base, "__lcb_records.json")
    if sidecar is None:
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return records
    if isinstance(payload, list):
        return payload
    return None


def _load_prompt_rollout_archive(base: Path) -> list[dict[str, Any]] | None:
    for candidate_suffix in (
        "__prompt_rollout_archive.jsonl.gz",
        "__prompt_rollout_archive.jsonl",
    ):
        candidate = _resolve_sibling(base, candidate_suffix)
        if candidate is not None:
            return list(_open_jsonl(candidate))
    return None


def _load_rollout_archive(base: Path) -> list[dict[str, Any]] | None:
    candidate = _resolve_sibling(base, "__rollout_archive.jsonl.gz")
    if candidate is None:
        candidate = _resolve_sibling(base, "__rollout_archive.jsonl")
    if candidate is None:
        return None
    return list(_open_jsonl(candidate))


def _load_prompt_profile(base: Path) -> list[dict[str, Any]] | None:
    for candidate_suffix in (
        "__prompt_profile.jsonl.gz",
        "__prompt_profile.jsonl",
    ):
        candidate = _resolve_sibling(base, candidate_suffix)
        if candidate is not None:
            return list(_open_jsonl(candidate))
    return None


def _coerce_rollout_index(record: dict[str, Any], fallback: int) -> int:
    for key in ("rollout_index", "generation_index"):
        value = record.get(key)
        if isinstance(value, int):
            return int(value)
    return int(fallback)


def _build_row_from_prompt_archive(
    archive_row: dict[str, Any],
    *,
    sample_id: int,
    lcb_records_by_qg: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    rollouts_raw = archive_row.get("rollouts") or []
    rollouts: list[dict[str, Any]] = []
    record_id = str(
        archive_row.get("record_id")
        or archive_row.get("question_id")
        or archive_row.get("sample_id", sample_id),
    )
    for idx, rollout in enumerate(rollouts_raw):
        rollout_index = _coerce_rollout_index(rollout, idx)
        completion_text = rollout.get("completion_text")
        completion_token_ids = rollout.get("completion_token_ids")
        grading: dict[str, Any] | None = None
        lcb_key = (record_id, rollout_index)
        lcb_record = lcb_records_by_qg.get(lcb_key)
        if lcb_record is not None and lcb_record.get("code_output") is not None:
            grading = {"code_output": lcb_record.get("code_output")}
        rollouts.append(
            {
                "rollout_index": rollout_index,
                "completion_text": completion_text,
                "completion_token_ids": (
                    [int(t) for t in completion_token_ids]
                    if isinstance(completion_token_ids, list)
                    else None
                ),
                "completion_token_count": rollout.get("length")
                or rollout.get("completion_token_count"),
                "total_token_count": rollout.get("total_token_count"),
                "finish_reason": rollout.get("finish_reason"),
                "loop_flag": bool(rollout.get("loop_flag", False)),
                "max_length_hit": bool(rollout.get("max_length_hit", False)),
                "first_loop_prefix_length": rollout.get("first_loop_prefix_length"),
                "loop_trigger": rollout.get("loop_trigger"),
                "length": rollout.get("length"),
                "relative_length": rollout.get("relative_length"),
                "cap_hit": rollout.get("cap_hit"),
                "tail_hit": rollout.get("tail_hit"),
                "grading": grading,
            }
        )
    row: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "sample_id": int(archive_row.get("sample_id", sample_id)),
        "split": archive_row.get("split") or archive_row.get("source_split"),
        "record_id": record_id,
        "record_metadata": archive_row.get("record_metadata") or {},
        "prompt": archive_row.get("prompt"),
        "prompt_token_ids": archive_row.get("prompt_token_ids"),
        "prompt_token_count": archive_row.get("prompt_token_count"),
        "effective_max_tokens": archive_row.get("effective_max_tokens"),
        "max_model_len": archive_row.get("max_model_len"),
        "prompt_too_long": bool(archive_row.get("prompt_too_long", False)),
        "prompt_profile": {
            key: archive_row.get(key)
            for key in (
                "p_cap",
                "p_loop",
                "loop_budget_share",
                "mu_log_rel",
                "mean_length",
                "mean_relative_length",
                "tail_threshold",
                "tail_hit_count",
                "majority_tail",
                "num_rollouts",
            )
            if key in archive_row
        }
        or None,
        "rollouts": rollouts,
    }
    if row["prompt"] is None or row["prompt_token_ids"] is None:
        row["degraded"] = "no_raw_prompt"
    if rollouts and all(r.get("completion_text") is None for r in rollouts):
        existing = row.get("degraded")
        marker = "no_raw_completion_text"
        row["degraded"] = (
            f"{existing},{marker}" if existing and existing != marker else marker
        )
    return row


def _build_rows_from_lcb_only(
    lcb_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct per-prompt rows from a bare ``__lcb_records.json`` list.

    Used when no prompt/rollout archive exists next to the legacy JSON (e.g.
    the March v2 LCB bundle). Groups records by ``question_id`` in first-seen
    order and emits ``rollout_bundle.v1`` rows with every raw-text field set to
    ``None`` and ``degraded`` = ``no_raw_completion_text``.
    """
    grouped: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for record in lcb_records:
        qid = str(record.get("question_id"))
        grouped.setdefault(qid, []).append(record)

    rows: list[dict[str, Any]] = []
    for sample_id, (question_id, records) in enumerate(grouped.items()):
        records_sorted = sorted(
            records,
            key=lambda r: int(r.get("generation_index", 0)),
        )
        prompt_too_long = bool(
            records_sorted and records_sorted[0].get("prompt_too_long")
        )
        rollouts: list[dict[str, Any]] = []
        for idx, record in enumerate(records_sorted):
            rollout_index = _coerce_rollout_index(record, idx)
            grading: dict[str, Any] = {}
            code_output = record.get("code_output")
            if code_output is not None:
                grading["code_output"] = code_output
            rollouts.append(
                {
                    "rollout_index": rollout_index,
                    "completion_text": None,
                    "completion_token_ids": None,
                    "completion_token_count": record.get("token_count"),
                    "total_token_count": record.get("total_token_count"),
                    "finish_reason": record.get("finish_reason"),
                    "loop_flag": bool(record.get("loop_flag", False)),
                    "max_length_hit": bool(record.get("max_length_hit", False)),
                    "first_loop_prefix_length": record.get(
                        "first_loop_prefix_length"
                    ),
                    "loop_trigger": None,
                    "length": record.get("token_count"),
                    "relative_length": None,
                    "cap_hit": None,
                    "tail_hit": None,
                    "grading": grading or None,
                }
            )
        first_record = records_sorted[0] if records_sorted else {}
        row = {
            "schema": BUNDLE_SCHEMA,
            "sample_id": sample_id,
            "split": None,
            "record_id": question_id,
            "record_metadata": {},
            "prompt": None,
            "prompt_token_ids": None,
            "prompt_token_count": first_record.get("prompt_token_count"),
            "effective_max_tokens": first_record.get("effective_max_tokens"),
            "max_model_len": first_record.get("max_model_len"),
            "prompt_too_long": prompt_too_long,
            "prompt_profile": None,
            "rollouts": [] if prompt_too_long else rollouts,
            "degraded": "no_raw_completion_text,no_raw_prompt",
        }
        rows.append(row)
    return rows


def _migrate_one(base: Path, out_dir: Path, *, overwrite: bool) -> dict[str, Any]:
    stats_payload = json.loads(base.read_text())
    metadata = dict(stats_payload.get("metadata") or {})
    counts = dict(stats_payload.get("counts") or {})
    metrics = dict(stats_payload.get("metrics") or {})

    archive_rows = _load_prompt_rollout_archive(base)
    lcb_records = _load_lcb_records(base)
    _ = _load_rollout_archive(base)
    _ = _load_prompt_profile(base)

    lcb_by_qg: dict[tuple[str, int], dict[str, Any]] = {}
    if lcb_records is not None:
        for record in lcb_records:
            key = (
                str(record.get("question_id")),
                int(record.get("generation_index", 0)),
            )
            lcb_by_qg[key] = record

    if archive_rows:
        rows = [
            _build_row_from_prompt_archive(
                archive_row,
                sample_id=int(archive_row.get("sample_id", idx)),
                lcb_records_by_qg=lcb_by_qg,
            )
            for idx, archive_row in enumerate(archive_rows)
        ]
    elif lcb_records is not None:
        rows = _build_rows_from_lcb_only(lcb_records)
    else:
        rows = []

    sidecar_path_str, bundle_path_str = bundle_paths(str(out_dir / base.stem))
    sidecar_path = Path(sidecar_path_str)
    bundle_path = Path(bundle_path_str)
    if bundle_path.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing bundle {bundle_path}; pass --overwrite."
        )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()

    with BundleWriter(str(bundle_path)) as writer:
        for row in rows:
            writer.write_row(row)

    metadata["bundle_file"] = bundle_path.name
    metadata.setdefault("stats_contract_version", "rollout_stats_v2")
    metadata["migrated_from_legacy"] = True
    sidecar_payload = {
        "schema": BUNDLE_SCHEMA,
        "metadata": metadata,
        "counts": counts,
        "metrics": metrics,
    }
    write_bundle_sidecar(str(sidecar_path), sidecar_payload)

    return {
        "base": base.name,
        "bundle": str(bundle_path),
        "sidecar": str(sidecar_path),
        "rows": len(rows),
        "degraded_rows": sum(1 for row in rows if row.get("degraded")),
    }


def main() -> None:
    args = _parse_args()
    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    if not in_dir.is_dir():
        raise SystemExit(f"--in-dir does not exist: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bases = _iter_legacy_bases(in_dir)
    if args.only is not None:
        keep = {name.strip() for name in args.only.split(",") if name.strip()}
        bases = [b for b in bases if b.stem in keep]
    if not bases:
        raise SystemExit(f"No legacy stats JSONs found under {in_dir}.")

    results: list[dict[str, Any]] = []
    for base in bases:
        print(f"[migrate] {base.name}", flush=True)
        result = _migrate_one(base, out_dir, overwrite=args.overwrite)
        results.append(result)
        print(
            f"  -> {result['bundle']} ({result['rows']} rows, "
            f"{result['degraded_rows']} degraded)",
            flush=True,
        )

    print(json.dumps({"migrated": results}, indent=2))


if __name__ == "__main__":
    main()
