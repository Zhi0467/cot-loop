#!/usr/bin/env python3
"""Validate the screened prompt-profile RFM registry against on-disk prompt-profile archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from steer.prompt_profile_rfm_stage_registry import validate_stage_registry


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
            "Validate excluded datasets such as AIME or sub-threshold benchmark surfaces "
            "in addition to the active screened-in set."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for the validation record.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_stage_registry(
        args.archive_source_root,
        include_excluded=args.include_excluded,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
