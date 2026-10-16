"""Individual dataset generator for the Musanze Harvest Intelligence Service.

Usage (from the project root):
    python data/generate_data.py --student-id ug2527104@ines.ac.rw --seed 7104

The same script recreates the submitted CSV byte-for-byte on a clean computer.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, DATA_DIR, student_digits, verification_code  # noqa: E402
from src.validation import validate_seed  # noqa: E402


def base_row_count(seed: int) -> int:
    """Base rows = 240 + (LLLL mod 41)."""
    return 240 + (seed % 41)


def generate_dataframe(seed: int) -> pd.DataFrame:
    """Generate the raw dataset (with deliberate missing values and duplicates)."""
    rng = np.random.default_rng(seed)
    n = base_row_count(seed)

    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fert = rng.uniform(0, 250, n)
    size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], n)
    support = rng.integers(0, 2, n)
    noise = rng.normal(0.0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil
        + 0.02 * fert
        + 1.30 * size
        - 0.25 * np.abs(temp - 18)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)

    df = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "soil_moisture_pct": soil,
            "fertilizer_kg": fert,
            "field_size_ha": size,
            "season": season,
            "cooperative_support": support,
            "yield_tons": yield_tons,
        }
    )
    yield_per_ha = df["yield_tons"] / df["field_size_ha"]
    df["high_risk"] = (yield_per_ha < yield_per_ha.median()).astype(int)

    # Deliberate data problems: m missing values in two columns, then 3 exact duplicates.
    m = round(0.05 * n)
    soil_pos = rng.choice(n, size=m, replace=False)
    fert_pos = rng.choice(n, size=m, replace=False)
    df.iloc[soil_pos, df.columns.get_loc("soil_moisture_pct")] = np.nan
    df.iloc[fert_pos, df.columns.get_loc("fertilizer_kg")] = np.nan

    dup_pos = rng.choice(n, size=3, replace=False)
    df = pd.concat([df, df.iloc[dup_pos]], ignore_index=True)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV with 4 decimal places and a fixed line ending (OS-independent bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")


def sha256_of_file(path: Path) -> str:
    """SHA-256 of the exact CSV bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fingerprint(csv_path: Path, seed: int, rows: int, out_path: Path) -> str:
    """Write artifacts/data_fingerprint.txt and return the hash."""
    digest = sha256_of_file(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"SHA256: {digest}\nFile: {csv_path.name}\nRows: {rows}\n"
        f"Seed: {seed}\nVerification code: {verification_code(seed)}\n",
        encoding="utf-8",
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the individual Musanze dataset.")
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    validate_seed(args.student_id, args.seed)
    regdigits = student_digits(args.student_id)
    csv_path = DATA_DIR / f"AI_A2_{regdigits}.csv"
    df = generate_dataframe(args.seed)
    write_csv(df, csv_path)
    digest = write_fingerprint(csv_path, args.seed, len(df), ARTIFACTS_DIR / "data_fingerprint.txt")
    print(f"[{verification_code(args.seed)}] wrote {csv_path} ({len(df)} rows)")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
