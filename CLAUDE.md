# CLAUDE.md

This file records how AI assistance was used while working on the ETA challenge.

## Tooling use

I used AI assistance for quick iteration, mainly to:

- sketch experiment scripts for route/time aggregation ideas
- compare smoothing and calibration variants quickly
- refactor the training code into a repeatable script
- check the Docker submission path and README coverage

The final inference path is deterministic and self-contained. It does not call any AI service or external API.

## Final model notes

The submitted model is a log-duration hierarchical lookup:

- route + hour + day-of-week table
- route + hour + month table
- route + hour and route-pair backoff priors
- count-weighted blend between day-of-week and month estimates

The trained artifact stores compact lookup arrays so the repo can be pushed without Git LFS while still staying comfortably inside the Docker size limit.
