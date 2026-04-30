# CLAUDE.md

This file documents AI-tooling usage for the challenge.

## How AI tooling was used

- Rapid experiment generation for feature engineering and smoothing variants.
- Fast iteration on training/evaluation scripts.
- Refactoring and packaging checks for the Dockerized submission surface.

## Final model summary

- Hierarchical zone/time lookup model trained in log-duration space.
- Smoothed backoff chain:
  - `(pickup, dropoff, hour, dow)` -> `(pickup, dropoff, hour)` -> `(pickup, dropoff)` -> pickup/dropoff marginals
- Final calibrated inference:
  - `prediction = exp(pred_log) * 0.985`
  - Clipped to `[30, 10800]` seconds.
