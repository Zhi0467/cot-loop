#!/usr/bin/env python3
"""Emit the prompt-profile RFM stage registry for bash/Slurm consumers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from steer.prompt_profile_rfm_stage_registry import stage_registry_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-source-root",
        default=None,
        help="Override the base archive root used to resolve data directories.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help=(
            "Emit excluded datasets such as AIME or sub-threshold benchmark surfaces "
            "in addition to the active screened-in set."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "tsv"),
        default="json",
        help="JSON is human-readable; TSV is suitable for bash loops.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = stage_registry_payload(
        args.archive_source_root,
        include_excluded=args.include_excluded,
    )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for dataset in payload["datasets"]:
        print(
            "\t".join(
                [
                    str(dataset["key"]),
                    str(dataset["archive_data_dir"]),
                    str(dataset["active_stage"]).lower(),
                ]
            )
        )


if __name__ == "__main__":
    main()
