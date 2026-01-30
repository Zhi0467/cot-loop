#!/usr/bin/env python3
"""Build AIME 2024/2025 JSONL from HF datasets.

Loads:
- math-ai/aime24 (split: test)
- math-ai/aime25 (split: test)
Uses only the `problem` column as `question`.
"""

import argparse
import json
from datasets import load_dataset


def load_problems(repo_id: str, split: str):
    ds = load_dataset(repo_id, split=split)
    if "problem" not in ds.column_names:
        raise ValueError(f"Missing 'problem' column in {repo_id}:{split} (columns={ds.column_names})")
    return [row["problem"] for row in ds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/aime_2024_2025.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    aime24 = load_problems("math-ai/aime24", args.split)
    aime25 = load_problems("math-ai/aime25", args.split)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, question in enumerate(aime24, start=1):
            row = {
                "id": f"AIME24-{i}",
                "year": 2024,
                "problem": i,
                "question": question,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for i, question in enumerate(aime25, start=1):
            row = {
                "id": f"AIME25-{i}",
                "year": 2025,
                "problem": i,
                "question": question,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
