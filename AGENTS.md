# AGENTS.md

Notes for anyone running or reviewing this submission.

## Scope

This repository is for the Gobblecube ETA challenge only. It does not include the Crossing challenge starter.

## Runtime contract

- `predict.py` exposes `predict(request: dict) -> float`
- `model.pkl` is loaded locally at import time
- prediction is deterministic
- inference makes no network calls

## Grader path

The Docker entrypoint runs:

```bash
python grade.py <input.parquet> <output.csv>
```

The output CSV contains `row_idx` and `prediction`, matching the starter grader contract.
