#!/usr/bin/env python
"""Train a fast hierarchical lookup ETA model.

This version models log-duration and exponentiates at inference. For skewed
travel-time distributions this behaves closer to a conditional median, which is
better aligned with MAE than raw means.
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ZONE_MAX = 265
ZONE_DIM = ZONE_MAX + 1
HOURS = 24
DOWS = 7


@dataclass(frozen=True)
class Smoothing:
    k_pair: float = 0.3
    k_pair_hour: float = 3.0
    k_pair_hour_dow: float = 5.0


@dataclass(frozen=True)
class Calibration:
    output_scale: float = 0.985
    clip_min: float = 30.0
    clip_max: float = 10_800.0


def _zero_stats() -> dict[str, np.ndarray | float | int]:
    pair_size = ZONE_DIM * ZONE_DIM
    pair_hour_size = pair_size * HOURS
    pair_hour_dow_size = pair_hour_size * DOWS

    return {
        "global_sum": 0.0,
        "global_count": 0,
        "pickup_sum": np.zeros(ZONE_DIM, dtype=np.float64),
        "pickup_count": np.zeros(ZONE_DIM, dtype=np.int64),
        "dropoff_sum": np.zeros(ZONE_DIM, dtype=np.float64),
        "dropoff_count": np.zeros(ZONE_DIM, dtype=np.int64),
        "pair_sum": np.zeros(pair_size, dtype=np.float64),
        "pair_count": np.zeros(pair_size, dtype=np.int64),
        "pair_hour_sum": np.zeros(pair_hour_size, dtype=np.float64),
        "pair_hour_count": np.zeros(pair_hour_size, dtype=np.int64),
        "pair_hour_dow_sum": np.zeros(pair_hour_dow_size, dtype=np.float64),
        "pair_hour_dow_count": np.zeros(pair_hour_dow_size, dtype=np.int64),
    }


def _safe_mean(sum_arr: np.ndarray, count_arr: np.ndarray, fallback: float) -> np.ndarray:
    out = np.full(sum_arr.shape, fallback, dtype=np.float64)
    np.divide(sum_arr, count_arr, out=out, where=count_arr > 0)
    return out


def _extract_time_parts(ts_str: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(ts_str, errors="coerce")
    if ts.isna().any():
        raise ValueError("Invalid timestamps found in requested_at.")
    hour = ts.dt.hour.to_numpy(dtype=np.int32)
    dow = ts.dt.dayofweek.to_numpy(dtype=np.int32)
    return hour, dow


def _transform_target(y_seconds: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log":
        return np.log(y_seconds)
    if mode == "raw":
        return y_seconds
    raise ValueError(f"Unsupported target mode: {mode}")


def _inverse_target(y_model: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log":
        return np.exp(y_model)
    if mode == "raw":
        return y_model
    raise ValueError(f"Unsupported target mode: {mode}")


def _accumulate_batch(
    df: pd.DataFrame,
    stats: dict[str, np.ndarray | float | int],
    target_mode: str,
) -> None:
    pu = df["pickup_zone"].to_numpy(dtype=np.int32, copy=False)
    do = df["dropoff_zone"].to_numpy(dtype=np.int32, copy=False)
    y_seconds = df["duration_seconds"].to_numpy(dtype=np.float64, copy=False)
    y = _transform_target(y_seconds, mode=target_mode)
    hour, dow = _extract_time_parts(df["requested_at"])

    pair_idx = pu * ZONE_DIM + do
    pair_hour_idx = pair_idx * HOURS + hour
    pair_hour_dow_idx = pair_hour_idx * DOWS + dow

    pair_size = ZONE_DIM * ZONE_DIM
    pair_hour_size = pair_size * HOURS
    pair_hour_dow_size = pair_hour_size * DOWS

    stats["global_sum"] = float(stats["global_sum"]) + float(y.sum())
    stats["global_count"] = int(stats["global_count"]) + int(y.size)

    stats["pickup_sum"] += np.bincount(pu, weights=y, minlength=ZONE_DIM)
    stats["pickup_count"] += np.bincount(pu, minlength=ZONE_DIM)
    stats["dropoff_sum"] += np.bincount(do, weights=y, minlength=ZONE_DIM)
    stats["dropoff_count"] += np.bincount(do, minlength=ZONE_DIM)

    stats["pair_sum"] += np.bincount(pair_idx, weights=y, minlength=pair_size)
    stats["pair_count"] += np.bincount(pair_idx, minlength=pair_size)

    stats["pair_hour_sum"] += np.bincount(
        pair_hour_idx, weights=y, minlength=pair_hour_size
    )
    stats["pair_hour_count"] += np.bincount(pair_hour_idx, minlength=pair_hour_size)

    stats["pair_hour_dow_sum"] += np.bincount(
        pair_hour_dow_idx, weights=y, minlength=pair_hour_dow_size
    )
    stats["pair_hour_dow_count"] += np.bincount(
        pair_hour_dow_idx, minlength=pair_hour_dow_size
    )


def build_stats(
    train_path: Path,
    batch_size: int,
    target_mode: str,
) -> dict[str, np.ndarray | float | int]:
    stats = _zero_stats()
    pf = pq.ParquetFile(train_path)
    columns = [
        "pickup_zone",
        "dropoff_zone",
        "requested_at",
        "duration_seconds",
    ]
    total_rows = 0
    t0 = time.time()
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        df = batch.to_pandas()
        total_rows += len(df)
        _accumulate_batch(df, stats, target_mode=target_mode)
        if total_rows % 5_000_000 < batch_size:
            elapsed = time.time() - t0
            print(f"  processed {total_rows:,} rows in {elapsed:.0f}s")
    print(f"  finished aggregating {total_rows:,} rows")
    return stats


def build_lookup_table(
    stats: dict[str, np.ndarray | float | int],
    smoothing: Smoothing,
    calibration: Calibration,
    target_mode: str,
) -> dict:
    global_count = int(stats["global_count"])
    global_sum = float(stats["global_sum"])
    if global_count == 0:
        raise RuntimeError("No rows aggregated from training data.")
    global_center = global_sum / global_count

    pickup_mean = _safe_mean(
        stats["pickup_sum"],
        stats["pickup_count"],
        fallback=global_center,
    )
    dropoff_mean = _safe_mean(
        stats["dropoff_sum"],
        stats["dropoff_count"],
        fallback=global_center,
    )

    pair_sum = stats["pair_sum"].reshape(ZONE_DIM, ZONE_DIM)
    pair_count = stats["pair_count"].reshape(ZONE_DIM, ZONE_DIM)
    pair_base_prior = 0.5 * (
        pickup_mean.reshape(ZONE_DIM, 1) + dropoff_mean.reshape(1, ZONE_DIM)
    )
    pair_prior = (pair_sum + smoothing.k_pair * pair_base_prior) / (
        pair_count + smoothing.k_pair
    )

    pair_hour_sum = stats["pair_hour_sum"].reshape(ZONE_DIM, ZONE_DIM, HOURS)
    pair_hour_count = stats["pair_hour_count"].reshape(ZONE_DIM, ZONE_DIM, HOURS)
    pair_hour_prior = (pair_hour_sum + smoothing.k_pair_hour * pair_prior[..., None]) / (
        pair_hour_count + smoothing.k_pair_hour
    )

    pair_hour_dow_sum = stats["pair_hour_dow_sum"].reshape(ZONE_DIM, ZONE_DIM, HOURS, DOWS)
    pair_hour_dow_count = stats["pair_hour_dow_count"].reshape(
        ZONE_DIM, ZONE_DIM, HOURS, DOWS
    )
    table = (pair_hour_dow_sum + smoothing.k_pair_hour_dow * pair_hour_prior[..., None]) / (
        pair_hour_dow_count + smoothing.k_pair_hour_dow
    )

    table[0, :, :, :] = global_center
    table[:, 0, :, :] = global_center
    table = table.astype(np.float32)

    global_duration = float(
        np.clip(
            _inverse_target(np.array([global_center], dtype=np.float64), mode=target_mode)[0]
            * calibration.output_scale,
            calibration.clip_min,
            calibration.clip_max,
        )
    )

    return {
        "version": 2,
        "target_mode": target_mode,
        "output_scale": float(calibration.output_scale),
        "global_center": float(global_center),
        "global_prediction": global_duration,
        "table": table,
        "clip_min": float(calibration.clip_min),
        "clip_max": float(calibration.clip_max),
        "zone_max": ZONE_MAX,
        "smoothing": {
            "k_pair": smoothing.k_pair,
            "k_pair_hour": smoothing.k_pair_hour,
            "k_pair_hour_dow": smoothing.k_pair_hour_dow,
        },
    }


def _predict_array(model: dict, pu: np.ndarray, do: np.ndarray, hour: np.ndarray, dow: np.ndarray) -> np.ndarray:
    raw = model["table"][pu, do, hour, dow].astype(np.float64, copy=False)
    target_mode = model.get("target_mode", "raw")
    pred = _inverse_target(raw, mode=target_mode)
    pred = pred * float(model.get("output_scale", 1.0))
    pred = np.clip(pred, float(model["clip_min"]), float(model["clip_max"]))
    return pred


def evaluate_dev(model: dict, dev_path: Path) -> float:
    dev = pd.read_parquet(
        dev_path,
        columns=[
            "pickup_zone",
            "dropoff_zone",
            "requested_at",
            "duration_seconds",
        ],
    )
    hour, dow = _extract_time_parts(dev["requested_at"])
    pu = dev["pickup_zone"].to_numpy(dtype=np.int32, copy=False)
    do = dev["dropoff_zone"].to_numpy(dtype=np.int32, copy=False)
    y = dev["duration_seconds"].to_numpy(dtype=np.float64, copy=False)

    preds = _predict_array(model, pu=pu, do=do, hour=hour, dow=dow)
    mae = float(np.mean(np.abs(preds - y)))
    return mae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ETA lookup model.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/train.parquet"),
        help="Path to training parquet",
    )
    parser.add_argument(
        "--dev-path",
        type=Path,
        default=Path("data/dev.parquet"),
        help="Path to dev parquet for local MAE check",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("submission/model.pkl"),
        help="Output model path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000_000,
        help="Parquet batch size used during aggregation",
    )
    parser.add_argument(
        "--target-mode",
        choices=["log", "raw"],
        default="log",
        help="Optimization space for the hierarchical table",
    )
    parser.add_argument("--k-pair", type=float, default=0.3)
    parser.add_argument("--k-pair-hour", type=float, default=3.0)
    parser.add_argument("--k-pair-hour-dow", type=float, default=5.0)
    parser.add_argument("--output-scale", type=float, default=0.985)
    parser.add_argument("--clip-min", type=float, default=30.0)
    parser.add_argument("--clip-max", type=float, default=10_800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoothing = Smoothing(
        k_pair=args.k_pair,
        k_pair_hour=args.k_pair_hour,
        k_pair_hour_dow=args.k_pair_hour_dow,
    )
    calibration = Calibration(
        output_scale=args.output_scale,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    if not args.train_path.exists():
        raise SystemExit(f"Missing {args.train_path}. Run data/download_data.py first.")
    if not args.dev_path.exists():
        raise SystemExit(f"Missing {args.dev_path}. Run data/download_data.py first.")

    print("Building aggregate statistics...")
    t0 = time.time()
    stats = build_stats(
        args.train_path,
        batch_size=args.batch_size,
        target_mode=args.target_mode,
    )
    print(f"Aggregations complete in {time.time() - t0:.1f}s")

    print("Building lookup table...")
    model = build_lookup_table(
        stats,
        smoothing=smoothing,
        calibration=calibration,
        target_mode=args.target_mode,
    )

    print("Evaluating on dev split...")
    mae = evaluate_dev(model, args.dev_path)
    model["dev_mae"] = mae
    print(f"Dev MAE: {mae:.3f} seconds")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
