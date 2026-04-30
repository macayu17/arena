# ETA Challenge Submission

This repo is my final submission for the **Gobblecube ETA Challenge**.

I focused on one goal: **beat the baseline with a model that is simple, fast, and robust in Docker grading**.

---

## Final score

- **Dev MAE (full dev set): 256.1 seconds**
- **Dev MAE (50k sample via `python grade.py`): 258.8 seconds**

For context, this is a large improvement over the provided GBT baseline (~351 s on Dev).

---

## What I shipped

This repository contains the required ETA submission surface:

- `predict.py` with `predict(request: dict) -> float`
- `Dockerfile` for sandbox grading
- `model.pkl` trained model artifact
- `README.md` (this write-up)

And the files needed to run the Docker grading path end-to-end:

- `grade.py`
- `requirements.txt`

Also included (as requested in the main take-home README):

- `AGENTS.md`
- `CLAUDE.md`

---

## Approach (what I built)

I ended up with a **hierarchical historical lookup model in log-duration space**, not a neural net and not a single global regressor.

At training time, I aggregate `log(duration_seconds)` at three levels:

1. `(pickup_zone, dropoff_zone, hour, day_of_week)`
2. `(pickup_zone, dropoff_zone, hour)`
3. `(pickup_zone, dropoff_zone)`

Then I smooth each level into the broader level beneath it:

- `pair-hour-dow` backs off to `pair-hour`
- `pair-hour` backs off to `pair`
- `pair` backs off to pickup/dropoff marginal priors

Final tuned smoothing values:

- `k_pair = 0.3`
- `k_pair_hour = 3.0`
- `k_pair_hour_dow = 5.0`

At inference:

1. Lookup the smoothed log prediction
2. Convert with `exp(pred_log)`
3. Apply a scalar calibration `* 0.985`
4. Clip to `[30, 10800]` seconds

Why this worked better than my early attempts:

- The target is heavy-tailed, and raw means are too sensitive to outliers.
- Log-space + hierarchical backoff behaved closer to a conditional median, which aligns better with MAE.
- The model keeps strong route/time locality while still handling sparse combinations cleanly.

---

## What I tried that did not work

I kept notes while iterating; these were the main dead ends:

- **Plain baseline-style XGBoost** with calendar + zone IDs: underfit route-time structure.
- **Recency weighting** (exponential half-life): made Dev MAE worse in this split.
- **30-minute slot granularity**: added variance without enough gain vs hourly bins.
- **Residual stacking with a booster on top of lookup predictions**: consistently degraded MAE.
- **Over-smoothing pair priors**: washed out useful route-specific signal.

---

## Inference constraints check

- **No external API calls at inference:** yes
- **CPU inference target (<= 200 ms/request):** yes (single lookup + timestamp parse)
- **Docker image size <= 2.5 GB:** yes (about 679 MB)

---

## Reproducibility

### Train model

```bash
python data/download_data.py
python train.py --out model.pkl
```

### Local scoring

```bash
python grade.py
```

### Docker grading path

```bash
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
```

---

## If I had more time

My next two experiments would be:

1. Add weather + holiday features (especially for the winter eval slice behavior).
2. Introduce robust quantile-aware group estimators instead of approximating median via log-mean only.

---

## Time spent

Roughly one focused day including exploration, tuning, packaging, and Docker validation.
