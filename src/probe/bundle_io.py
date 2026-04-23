"""Rollout-bundle I/O helpers.

A "rollout bundle" is the unified artifact produced by
``scripts/rollout/collect_model_stats.py`` for a single run of a model on a dataset:

- ``<base>.jsonl.gz`` - one JSON line per prompt, carrying the prompt text,
  prompt token ids, task metadata, per-rollout completion text + token ids +
  flags + task-specific grading, and prompt-level aggregates.
- ``<base>.json`` - small sidecar with run metadata (generation config, task
  kind, release_version, lm_style, thinking_mode), aggregate counts, metrics,
  and ``lcb_native_metrics`` when applicable.

This module is the single source of truth for bundle paths, streaming writes,
reading, and the resume contract. Consumers should route through it rather
than opening the files directly.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from contextlib import AbstractContextManager
from typing import Any, Iterable, Iterator

BUNDLE_SCHEMA = "rollout_bundle.v1"
BUNDLE_SUFFIX = ".jsonl.gz"
SIDECAR_SUFFIX = ".json"


def bundle_paths(out_path: str | os.PathLike[str]) -> tuple[str, str]:
    """Return ``(sidecar_json_path, bundle_jsonl_gz_path)``.

    ``out_path`` may be the sidecar JSON path, the bundle path, or a bare base
    path (e.g. ``outputs/model_stats/foo``); all three resolve to the same pair.
    """
    out_path = os.fspath(out_path)
    if out_path.endswith(BUNDLE_SUFFIX):
        base = out_path[: -len(BUNDLE_SUFFIX)]
    elif out_path.endswith(".jsonl"):
        base = out_path[: -len(".jsonl")]
    elif out_path.endswith(SIDECAR_SUFFIX):
        base = out_path[: -len(SIDECAR_SUFFIX)]
    else:
        base = out_path
    return base + SIDECAR_SUFFIX, base + BUNDLE_SUFFIX


class BundleWriter(AbstractContextManager):
    """Append-only JSONL writer backed by a gzipped file.

    Each call to :meth:`write_row` appends one JSON line. Multiple opens of
    the same path simply concatenate gzip members; :func:`iter_bundle_rows`
    reads across members transparently. This makes the writer safe across
    crash/resume without a separate checkpoint file.
    """

    def __init__(self, bundle_path: str):
        self._path = bundle_path
        self._handle: Any | None = None
        out_dir = os.path.dirname(bundle_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    @property
    def path(self) -> str:
        return self._path

    def __enter__(self) -> "BundleWriter":
        self._handle = gzip.open(self._path, "at", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
            except Exception:
                pass
            self._handle.close()
            self._handle = None

    def write_row(self, row: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("BundleWriter.write_row called outside context.")
        json.dump(row, self._handle, ensure_ascii=False)
        self._handle.write("\n")
        self._handle.flush()


def iter_bundle_rows(bundle_path: str) -> Iterator[dict[str, Any]]:
    """Yield each JSON row from ``bundle_path`` (transparent across members)."""
    with gzip.open(bundle_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def read_bundle(bundle_path: str) -> list[dict[str, Any]]:
    """Materialize the full bundle into a list of rows."""
    return list(iter_bundle_rows(bundle_path))


def read_bundle_sidecar(sidecar_path: str) -> dict[str, Any]:
    with open(sidecar_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_bundle_sidecar(sidecar_path: str, payload: dict[str, Any]) -> None:
    out_dir = os.path.dirname(sidecar_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def completed_sample_ids(bundle_path: str) -> set[int]:
    """Return the set of sample_ids already present in ``bundle_path``.

    Used for resume: the caller skips samples whose ids appear in the returned
    set. Returns the empty set when the bundle file does not exist. A partially
    truncated bundle returns whatever prefix parses cleanly; the caller is
    expected to continue appending after that prefix.
    """
    if not os.path.isfile(bundle_path):
        return set()
    ids: set[int] = set()
    try:
        for row in iter_bundle_rows(bundle_path):
            sid = row.get("sample_id")
            if isinstance(sid, int):
                ids.add(int(sid))
    except (OSError, EOFError, json.JSONDecodeError):
        return ids
    return ids


def rewrite_bundle(bundle_path: str, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically rewrite ``bundle_path`` with ``rows``.

    Writes to a tmp file in the same directory, then renames into place.
    """
    dest_dir = os.path.dirname(bundle_path) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".bundle_tmp_",
        suffix=BUNDLE_SUFFIX,
        dir=dest_dir,
    )
    os.close(fd)
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
            for row in rows:
                json.dump(row, handle, ensure_ascii=False)
                handle.write("\n")
        os.replace(tmp_path, bundle_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def concat_bundles(sources: list[str], destination: str) -> None:
    """Concatenate several bundle files into ``destination``.

    Gzip allows raw concatenation of members. The sources are read as gzip
    streams (so partial/corrupt bytes are rejected), recompressed into a
    single output, and then unlinked. Empty / missing sources are skipped.
    """
    dest_dir = os.path.dirname(destination) or "."
    os.makedirs(dest_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".bundle_concat_",
        suffix=BUNDLE_SUFFIX,
        dir=dest_dir,
    )
    os.close(tmp_fd)
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as dest_handle:
            for source in sources:
                if not source or not os.path.isfile(source):
                    continue
                with gzip.open(source, "rt", encoding="utf-8") as src_handle:
                    for line in src_handle:
                        dest_handle.write(line)
        os.replace(tmp_path, destination)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    for source in sources:
        if not source or not os.path.isfile(source):
            continue
        if os.path.abspath(source) == os.path.abspath(destination):
            continue
        try:
            os.unlink(source)
        except OSError:
            pass


def iter_rollouts(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield each rollout dict from a bundle row (empty when prompt-too-long)."""
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list):
        return
    for rollout in rollouts:
        if isinstance(rollout, dict):
            yield rollout
