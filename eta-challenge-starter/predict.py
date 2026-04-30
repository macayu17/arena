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

_TABLE = _MODEL["table"]
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
        pred = _decode(float(_TABLE[pickup_zone, dropoff_zone, ts.hour, ts.weekday()]))

    return float(np.clip(pred, _CLIP_MIN, _CLIP_MAX))
