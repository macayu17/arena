# ETA Challenge Submission

This is my submission for the Gobblecube ETA Challenge. I picked the ETA track and kept the repo focused on that one problem.

The final model is deliberately simple at inference time: load one local pickle, parse the request timestamp, do a few array lookups, and return a duration in seconds. There are no network calls or external services in the prediction path.

## Final score

My local Dev results:

- Full Dev MAE: **253.8 seconds** (`253.803` before rounding)
- 50k sample MAE from `python grade.py`: **256.7 seconds**

The provided baseline is around 351 seconds on Dev, so this is mainly a feature/statistics improvement rather than a heavier modeling stack.

## Files in this repo

Required submission files:

- `predict.py`: exposes `predict(request: dict) -> float`
- `Dockerfile`: packages the submission for the grader
- `model.pkl`: trained model artifact
- `README.md`: this write-up

Supporting files I kept because they are useful for reproducing and checking the submission:

- `grade.py`: local scoring and grader-mode prediction writer
- `requirements.txt`: runtime dependencies
- `train.py`: training script that rebuilds `model.pkl`
- `data/download_data.py` and `data/schema.md`: data download/format references from the ETA starter
- `AGENTS.md` and `CLAUDE.md`: metadata requested by the main take-home README

## Model approach

I started with the baseline idea and quickly found that the model needed to respect route-specific history much more directly. A simple route-pair average was already stronger than the starter GBT, so I leaned into that instead of making the tree model larger.

The final model is a hierarchical lookup model trained on `log(duration_seconds)`.

I aggregate historical trips at these levels:

- `(pickup_zone, dropoff_zone, hour, day_of_week)`
- `(pickup_zone, dropoff_zone, hour, month)`
- `(pickup_zone, dropoff_zone, hour)`
- `(pickup_zone, dropoff_zone)`

The backoff chain is:

- route + hour + day-of-week backs off to route + hour
- route + hour + month backs off to route + hour
- route + hour backs off to route pair
- route pair backs off to pickup/dropoff marginal priors

At prediction time I blend the day-of-week and month estimates using confidence weights from the historical counts. Then I exponentiate back into seconds, apply one scalar calibration, and clip to the valid cleaned-data range.

Final tuned values:

- `k_pair = 0.3`
- `k_pair_hour = 3.3`
- `k_pair_hour_dow = 4.7`
- `k_pair_hour_month = 8.0`
- month blend multiplier `0.5`
- output scale `0.987`

I store the main lookup arrays as `float16` and quantize the month confidence weights. That keeps `model.pkl` below GitHub's 100 MB hard limit while preserving the score gain from the month-aware blend.

## Why this worked

Trip duration has a long tail, so raw conditional means are pulled around by unusual trips. Training in log space behaved closer to a conditional median, which is a better fit for MAE.

The time features also mattered. Hour captures normal traffic rhythm; day-of-week captures commute/weekend patterns; month helped on the Dev split because the holdout window is late December, where traffic is not quite the same as an all-year average.

## Things I tried that did not help

- A larger baseline-style XGBoost model on zone IDs and calendar fields. It underfit the route-specific structure.
- Pure route-pair averages. Strong for a tiny model, but missing hour/day patterns.
- 30-minute time slots. They added variance and did not beat hourly bins.
- Exponential recency weighting. It hurt Dev MAE for the split used here.
- A residual booster stacked on top of the lookup prediction. It overcorrected and made the result worse.

## Constraints check

- Inference is CPU-only and comfortably under 200 ms per request.
- Docker image is about 719 MB, under the 2.5 GB limit.
- Inference uses only local files and makes no external API calls.
- The model is trained only from the provided 2023 train split.

## Reproduce

From the repository root:

```bash
python data/download_data.py
python train.py --out model.pkl
python grade.py
```

Docker path:

```bash
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
```

## Next experiments

If I kept going, I would try holiday/weather joins first. The model already captures normal historical traffic patterns, so the next likely gains are from signals that explain unusual days rather than adding more model complexity.

I would also try a true approximate median/quantile sketch per group. The current log-mean approach is a practical approximation, but direct quantile aggregation may fit MAE better.

## Time spent

About one focused day, including data checks, model iterations, packaging, and Docker validation.
