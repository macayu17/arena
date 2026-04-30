# AGENTS.md

This repository was developed as an individual Gobblecube ETA challenge submission.

## Ownership

- Single-author submission (no collaborator contributions).

## Inference behavior

- `predict.py` is deterministic and self-contained.
- Inference uses only local artifacts (`model.pkl`) and standard Python libraries.
- No network or external API calls are made during prediction.

## Packaging

- Root-level `Dockerfile` is the submission entrypoint.
- Grader interface is `python grade.py <input.parquet> <output.csv>`.
