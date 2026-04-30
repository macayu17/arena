# ETA Challenge Submission

## Final score

Dev MAE (full dev via `python submission/train.py` evaluation): **256.1 s**  
Dev MAE (50k local sample via `python submission/grade.py`): **258.8 s**

## Approach

This submission uses a **hierarchical historical lookup model in log-duration space** instead of gradient boosting:

1. Aggregate historical `log(duration_seconds)` on:
   - `(pickup_zone, dropoff_zone)`
   - `(pickup_zone, dropoff_zone, hour)`
   - `(pickup_zone, dropoff_zone, hour, day_of_week)`
2. Apply smoothing so sparse groups back off gracefully:
   - pair-level estimates back off to pickup/dropoff marginals
   - pair-hour backs off to pair
   - pair-hour-dow backs off to pair-hour
   - tuned smoothing strengths: `k_pair=0.3`, `k_pair_hour=3.0`, `k_pair_hour_dow=5.0`
3. At inference, convert with `exp(pred_log)` and apply a calibrated global scale `0.985`, then clip to `[30, 10800]`.
4. Store the final dense lookup table in `model.pkl` and answer each request with one table lookup.

Using log-space improves robustness to long-tail durations and better approximates the conditional median, which aligns with MAE. Inference remains very fast (single CPU lookup plus timestamp parsing).

## What I tried that did not work as well

- The provided XGBoost baseline features (zone IDs + calendar fields) underfit this problem and scored much worse than high-granularity historical route-time statistics.
- Pure pair averages without hour/day context miss major traffic-pattern shifts.
- Over-smoothing pair-level estimates increased MAE by washing out route-specific signal.
- Recency weighting and residual XGBoost stacking both degraded Dev MAE versus the tuned hierarchical lookup.

## Where AI tooling helped most

- Rapidly iterated candidate feature hierarchies and fallback strategies.
- Produced fast experiment scripts to compare MAE across multiple smoothing setups.
- Refactored training into a memory-safe batch aggregation pipeline over Parquet data.

## Next experiments

- Add external weather/holiday features to capture systematic shocks not visible in request metadata.
- Add geometric features (zone-centroid distance and borough-level priors) and blend with this lookup model.
- Try approximate group-median estimators (e.g., quantile sketches) directly instead of log-mean approximation.

## Reproducibility

From repo root:

```bash
python data/download_data.py
python submission/train.py --out submission/model.pkl
python submission/grade.py
```

Docker packaging test:

```bash
cd submission
docker build -t my-eta-submission .
docker run --rm -v $(pwd)/../data:/work my-eta-submission /work/dev.parquet /work/preds.csv
```
