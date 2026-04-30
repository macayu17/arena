"""Submission interface for Gobblecube ETA challenge."""

from __future__ import annotations

import math
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)

_TABLE_DOW = _MODEL.get("table_dow", _MODEL["table"])
_TABLE_MONTH = _MODEL.get("table_month")
_DOW_WEIGHT = _MODEL.get("dow_weight")
_MONTH_WEIGHT = _MODEL.get("month_weight")
_MONTH_WEIGHT_U8 = _MODEL.get("month_weight_u8")
_MONTH_WEIGHT_SCALE = float(_MODEL.get("month_weight_scale", 1.0))
_CLIP_MIN = float(_MODEL.get("clip_min", 30.0))
_CLIP_MAX = float(_MODEL.get("clip_max", 10_800.0))
_ZONE_MAX = int(_MODEL.get("zone_max", 265))
_TARGET_MODE = str(_MODEL.get("target_mode", "raw"))
_OUTPUT_SCALE = float(_MODEL.get("output_scale", 1.0))

if "global_prediction" in _MODEL:
    _GLOBAL_PRED = float(_MODEL["global_prediction"])
else:
    _global_center = float(_MODEL.get("global_center", _MODEL.get("global_mean", 900.0)))
    if _TARGET_MODE == "log":
        _GLOBAL_PRED = float(math.exp(_global_center) * _OUTPUT_SCALE)
    else:
        _GLOBAL_PRED = float(_global_center * _OUTPUT_SCALE)


def _decode(raw_value: float) -> float:
    if _TARGET_MODE == "log":
        return float(math.exp(raw_value) * _OUTPUT_SCALE)
    return float(raw_value * _OUTPUT_SCALE)


def predict(request: dict) -> float:
    """Predict trip duration (seconds) for one request."""
    pickup_zone = int(request["pickup_zone"])
    dropoff_zone = int(request["dropoff_zone"])
    ts = datetime.fromisoformat(request["requested_at"])

    if (
        pickup_zone < 1
        or pickup_zone > _ZONE_MAX
        or dropoff_zone < 1
        or dropoff_zone > _ZONE_MAX
    ):
        pred = _GLOBAL_PRED
    else:
        hour = ts.hour
        dow = ts.weekday()
        raw_dow = float(_TABLE_DOW[pickup_zone, dropoff_zone, hour, dow])
        if _TABLE_MONTH is not None and _MONTH_WEIGHT_U8 is not None:
            month = ts.month - 1
            raw_month = float(_TABLE_MONTH[pickup_zone, dropoff_zone, hour, month])
            month_weight = (
                float(_MONTH_WEIGHT_U8[pickup_zone, dropoff_zone, hour, month])
                * _MONTH_WEIGHT_SCALE
            )
            raw = (raw_dow + month_weight * raw_month) / (1.0 + month_weight)
        elif _TABLE_MONTH is not None and _DOW_WEIGHT is not None and _MONTH_WEIGHT is not None:
            month = ts.month - 1
            raw_month = float(_TABLE_MONTH[pickup_zone, dropoff_zone, hour, month])
            dow_weight = float(_DOW_WEIGHT[pickup_zone, dropoff_zone, hour, dow])
            month_weight = float(_MONTH_WEIGHT[pickup_zone, dropoff_zone, hour, month])
            denom = dow_weight + month_weight
            if denom > 0:
                raw = (dow_weight * raw_dow + month_weight * raw_month) / denom
            else:
                raw = raw_dow
        else:
            raw = raw_dow
        pred = _decode(raw)

    return float(np.clip(pred, _CLIP_MIN, _CLIP_MAX))
