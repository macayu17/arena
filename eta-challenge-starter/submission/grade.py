#!/usr/bin/env python
"""Scoring harness for the submission package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict import predict

REQUEST_FIELDS = ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count"]
DEFAULT_DEV_PATH = Path(__file__).resolve().parent.parent / "data" / "dev.parquet"


def run(input_path: Path, output_path: Path | None, sample_n: int | None = None) -> None:
    df = pd.read_parquet(input_path)
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"Predicting {len(df):,} rows from {input_path.name}...", file=sys.stderr)

    preds = np.empty(len(df), dtype=np.float64)
    for i, req in enumerate(df[REQUEST_FIELDS].to_dict("records")):
        preds[i] = predict(req)

    if output_path is not None:
        if "row_idx" in df.columns:
            row_idx = df["row_idx"].to_numpy()
        else:
            row_idx = np.arange(len(df), dtype=np.int64)
        pd.DataFrame({"row_idx": row_idx, "prediction": preds}).to_csv(output_path, index=False)
        print(f"Wrote {len(preds):,} predictions to {output_path}", file=sys.stderr)
        return

    if "duration_seconds" not in df.columns:
        raise SystemExit("Local grading needs a `duration_seconds` column in the parquet.")
    mae = float(np.mean(np.abs(preds - df["duration_seconds"].to_numpy())))
    print(f"MAE: {mae:.1f} seconds")


def main(argv: list[str]) -> None:
    if len(argv) == 1:
        run(DEFAULT_DEV_PATH, None, sample_n=50_000)
    elif len(argv) == 3:
        run(Path(argv[1]), Path(argv[2]))
    else:
        print(
            "Usage:\n"
            "  python grade.py                              # local dev grading (50k sample)\n"
            "  python grade.py <input.parquet> <output.csv>  # grader mode",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
