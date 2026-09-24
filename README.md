# AdSignal — Data Platform

Ad campaign intelligence platform. This repository holds the Python data
platform: synthetic data generation, ETL, models, and the serving API.
The dashboard lives in the `portfolio-3d` repository.

## The data is synthetic

AdSignal generates its own ad-campaign data. No real advertiser, campaign,
or user data is used anywhere in this project. The generator injects
anomalies with known ground-truth causes, which is what makes supervised
training and honest evaluation possible.

## Quick start

```bash
uv sync
uv run dbt deps --project-dir dbt --profiles-dir dbt
uv run pytest
```

## Architecture

Raw events are written as Parquet and transformed by dbt against DuckDB.
Only aggregated daily marts are loaded into Postgres, which serves the API.
Bulk analytical scans and low-latency point reads have different access
patterns and belong in different stores.

`spend` and `revenue` are `float64` deliberately: they are aggregated
synthetic analytics metrics feeding a feature store, not ledger entries, and
float64 is the standard representation for spend in analytics warehouses.

`stg_creatives` and the advertiser dimension have no current consumer in
this repository — they exist for the planned dashboard's vertical filtering,
so their absence of a consumer today is intentional, not dead code.

A full architecture decision record (`ARCHITECTURE.md`) is planned but not
yet written.

## Running the stack

```bash
cp .env.example .env
docker compose up -d
```

Dagster is at http://localhost:3000. Materialize `catalog_and_raw_events`,
`dbt_marts`, and `campaign_daily_mart` in that order, or run the
`adsignal_daily` job which does all three.

Generating the full ten million events takes several minutes. For a fast
local loop, set a smaller target:

```bash
ADSIGNAL_TARGET_EVENTS=100000 docker compose up -d
```

## Pipeline stages

| Asset | What it does |
|---|---|
| `catalog_and_raw_events` | Generates the catalog, streams events to Parquet, records anomaly labels |
| `dbt_marts` | Runs `dbt build` against DuckDB: staging views, the fact table, and data tests |
| `campaign_daily_mart` | Full-refreshes the fact table into Postgres for the serving layer |

Every asset is idempotent. Re-running the job updates in place: the mart
load truncates the serving table and reinserts within a single transaction,
so it never duplicates rows and never leaves stale rows behind from a
previous run with a different dataset.

## Development

```bash
uv sync
uv run dbt deps --project-dir dbt --profiles-dir dbt
uv run ruff check . && uv run mypy && uv run pytest
```

Tests that need Postgres skip automatically when it is not reachable. Start
it with `docker compose up -d postgres`.
