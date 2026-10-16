"""CLI prediction: python predict.py --json samples/valid.json"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src.inference import ModelsNotFoundError, predict_record
from src.schemas import format_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict yield, risk and cluster for one farm record.")
    parser.add_argument("--json", required=True, help="Path to a JSON file with one record")
    args = parser.parse_args()
    try:
        record = json.loads(Path(args.json).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(json.dumps({"error": f"file not found: {args.json}"}), file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid JSON: {exc}"}), file=sys.stderr)
        return 2
    try:
        result = predict_record(record)
    except ValidationError as exc:
        print(json.dumps({"error": "validation_error", "detail": format_errors(exc.errors())}, indent=2), file=sys.stderr)
        return 2
    except ModelsNotFoundError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
