"""Single entry point: python run_all.py --student-id ug2527104@ines.ac.rw --seed 7104"""
from __future__ import annotations

import argparse
import sys

from src.pipeline import run_pipeline
from src.validation import DataValidationError


def main() -> int:
    parser = argparse.ArgumentParser(description="Musanze Harvest Intelligence Service - full pipeline")
    parser.add_argument("--student-id", required=True, help="Institutional username, e.g. ug2527104@ines.ac.rw")
    parser.add_argument("--seed", type=int, required=True, help="Last four digits of the registration number")
    args = parser.parse_args()
    try:
        run_pipeline(args.student_id, args.seed)
    except (DataValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
