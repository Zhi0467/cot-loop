from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    config: str | None
    split: str
    max_samples: int | None = None


@dataclass(frozen=True)
class SampleRecord:
    sample_id: int
    prompt: str
    source_split: str
