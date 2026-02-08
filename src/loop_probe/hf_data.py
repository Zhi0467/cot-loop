import random
from collections.abc import Sequence
from dataclasses import asdict

from datasets import load_dataset

from .types import DatasetSpec, SampleRecord


def specs_equal(a: DatasetSpec, b: DatasetSpec) -> bool:
    return asdict(a) == asdict(b)


def load_prompt_records(spec: DatasetSpec, prompt_field: str) -> list[SampleRecord]:
    ds = load_dataset(spec.dataset, spec.config, split=spec.split)
    if prompt_field not in ds.column_names:
        raise SystemExit(
            f"Prompt field '{prompt_field}' not found in dataset columns: {ds.column_names}"
        )

    if spec.max_samples is not None:
        if spec.max_samples < 1:
            raise SystemExit("--*-max-samples must be >= 1 when provided.")
        limit = min(len(ds), spec.max_samples)
        ds = ds.select(range(limit))

    records: list[SampleRecord] = []
    for idx, row in enumerate(ds):
        prompt = row[prompt_field]
        if prompt is None:
            continue
        records.append(
            SampleRecord(sample_id=idx, prompt=str(prompt), source_split=spec.split)
        )

    return records


def split_records(
    records: Sequence[SampleRecord],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not 0.0 < test_ratio < 1.0:
        raise SystemExit("test_ratio must be in (0, 1).")

    work = list(records)
    if len(work) < 2:
        raise SystemExit("Need at least 2 rows to split train/test from a single dataset.")

    rng = random.Random(seed)
    rng.shuffle(work)

    test_size = max(1, int(round(len(work) * test_ratio)))
    if test_size >= len(work):
        test_size = len(work) - 1

    test_records = work[:test_size]
    train_records = work[test_size:]
    return train_records, test_records
