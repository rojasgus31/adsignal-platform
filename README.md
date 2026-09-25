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

## Models

| Model | Purpose | Framework |
|---|---|---|
| IsolationForest | flag campaign-days that depart from the campaign's own recent behaviour | scikit-learn |
| HistGradientBoostingClassifier | predict an anomaly's cause, severity and whether to escalate | scikit-learn |

Features are rolling ratios and z-scores computed over a window that **ends the
day before** the row it describes. A baseline including the current day would
let the model see the anomaly it is meant to predict.

The train/holdout split is by date, never random: a random split over a time
series trains on the future to predict the past.

### What these models do and do not show

`anomaly_cause` is assigned by the synthetic generator from deterministic
effect multipliers, so the classifier recovers an encoded signal rather than
discovering one. What this demonstrates is the engineering around a model —
feature design that avoids leakage, temporal evaluation, versioned training
runs, a CI quality gate, and drift monitoring in production — not a research
result.

### Monitoring

`drift_reports` records a Population Stability Index and a two-sample
Kolmogorov–Smirnov test per feature on every run. PSI above 0.2 or a KS
p-value below 0.01 marks a feature as drifted. Implemented directly rather
than via a monitoring library: it is forty lines and every number is
explainable.

### Quality gate

`.github/workflows/eval.yml` trains on a fresh mart and fails the build when
triage macro F1 falls below the floor in `adsignal/ml/evaluation.py`, or when
the detector's ROC AUC falls below its own floor. The detector is a hard
pre-filter ahead of the classifier — the scoring asset only triages the rows
the detector flags — so a detector whose signal has degraded can silently
starve the classifier of real anomalies while triage metrics still look
healthy. AUC gates the detector rather than recall: recall is threshold- and
base-rate-dependent (contamination, and therefore the flagging threshold, is
derived from the training data's own anomaly rate), so a fixed recall floor
moves whenever the dataset's anomaly rate moves and is not a stable
regression gate. AUC asks a base-rate-independent question instead — does the
detector's score still separate anomalies from normal days at all. Both
gates must pass for the build to succeed; a regression on either cannot reach
`main` unnoticed.

The detector's contamination is derived from the training window's own
observed anomaly rate, which a sweep on real pipeline data confirmed
maximises F1 — this is an **F1-optimal pre-filter, not a recall-oriented
one**. At a ~5% base rate that costs about two thirds of recall (0.31
measured against 0.35 precision): most real anomalies never reach the
classifier at all. That is the trade accepted, because a lower-precision
pre-filter would make the downstream triage output unusable.

Because the two models are evaluated separately, neither number answers what
a user of the system actually experiences. `evaluate_pipeline` (also printed
by `gate.py`, not yet gated on) runs the real path — the classifier scoring
whatever the detector flagged, true anomaly or not — and reports it: on the
live holdout the detector flags 85 rows, of which only 30 (35%) are true
anomalies; and 22 of the 37 rows marked `escalate=True` (60%) are false
alarms. The headline triage macro F1 (0.8118) is measured on the
classifier's best case (true anomalies only) — the end-to-end numbers are
the honest picture of what the pipeline pages a human about. Exact figures
move slightly run to run as the models retrain; run `uv run python -m
adsignal.cli.quality_gate` for the current ones.

### Model features

`min_samples_leaf` for both triage models is a named constant
(`adsignal.ml.triage._MIN_SAMPLES_LEAF`), picked from a sweep against the
real mart's temporal holdout rather than tuned to a synthetic test fixture:
5 → 0.8119, 10 → 0.8118, 20 → 0.7777, 30 → 0.7907, 50 → 0.8358 macro F1. 50
scores highest on that single holdout split, but only values below 10 still
learn per-row signal from a cold-start-sized training window (~15 labelled
anomalies) instead of collapsing to the class prior — see
`test_small_training_window_still_learns_per_row_signal`. 5 and 10 are
statistically indistinguishable on the real mart, so 5 is kept.

`channel` is fed to the triage classifier as a native categorical
(`HistGradientBoostingClassifier`/`Regressor` support `categorical_features`
directly), delivering the categorical feature named in the spec rather than
leaving it declared but unused.

## Serving API

`src/adsignal/api/` is a read-only FastAPI service over the tables the
pipeline writes.

| Endpoint | Returns |
|---|---|
| `GET /health` | Liveness check |
| `GET /campaigns` | Every known campaign |
| `GET /campaigns/{id}/metrics` | One campaign's full daily time series |
| `GET /anomalies` | Every current anomaly prediction |
| `GET /anomalies/{campaign_id}/{event_date}` | One campaign-day's prediction |
| `GET /models` | Every training run's evaluation, newest first |
| `GET /benchmark` | Every run plus which narrative provider answered most recently |
| `POST /search` | Semantic search over stored entity embeddings |
| `POST /ask` | A natural-language question, answered from precomputed data |

Run it with `docker compose up -d api`, or locally with
`uv run uvicorn adsignal.api.app:create_app --factory --port 8000`. The
router layer issues no SQL of its own: every endpoint reads through a
`warehouse/` reader, and each of those creates its tables if they do not
exist yet, so a deployment where the pipeline has never run answers with
empty results rather than an error.

### Precomputation rule

Every endpoint except `/ask` reads rows the pipeline already computed and
stored — visitor traffic against them triggers no inference and no
external call, regardless of load. `/ask` is the sole exception: it is
the only path that does live work per request, which is exactly why it
is the only one that is rate-limited and cached (below).

### The narrative provider chain

`narrative/build.py` assembles a chain of providers in a fixed order:
Gemini, then Groq, then Jev, then Anthropic — each included only if its
API key is configured — with a `StubProvider` always last. Free tiers
are tried first, the paid provider last, and the stub guarantees a
deployment with no keys at all still renders a narrative (placeholder
text) rather than erroring. Each provider sits behind its own circuit
breaker in `NarrativeChain`; a provider that has failed enough times in
a row is skipped without being called until its cooldown elapses, so
one dead backend costs nothing on later requests.

### `/ask`: named templates, not generated SQL

The spec describes `/ask` as "natural language to SQL". This
implementation deliberately does **not** generate SQL. A keyword
matcher (or, in principle, a model) resolves the question to the
*name* of one of a fixed, closed set of query templates declared in
`adsignal/api/queries.py` — never to a query string. That name is
validated against the known template set before anything runs, and
rejected with a 400 if it does not match one exactly. Each template's
own parameters (for example, a result `limit`) are validated and bound
with `%s` placeholders; nothing a request supplies — not the template
name, not a parameter, not a column, not an ORDER BY direction — is
ever interpolated into SQL. A component that can only ever choose among
a handful of known names cannot emit an injection: this is not
SQL-injection sanitisation applied after the fact, it is a data path
where no such string can exist in the first place.

Because `/ask` is the one endpoint that does live work per request, it
is guarded by a per-IP `TokenBucket` (`adsignal/api/ratelimit.py`) — a
token bucket refilled gradually over time, returning 429 once a
caller's budget is exhausted — and identical questions are cached for a
short TTL so a repeated question is answered once, not recomputed on
every call. The rate limiter is checked on every request regardless of
a cache hit, so a burst of one repeated question still trips the
limiter; the cache only saves the downstream template resolution and
database round-trip. Both are per-app collaborators built in
`create_app` and injected through `api/deps.py`, and the cache is
bounded (LRU, hard capacity) as well as TTL'd, because its key is
caller-supplied free text.

**Known gap:** the plan describes resolving a question to a template
*via the narrative chain* when a provider key is configured. The
existing `NarrativeProvider` contract is
`generate(request: NarrativeRequest) -> str`, where `NarrativeRequest`
is a closed dataclass of precomputed statistics (campaign/anomaly
counts, top causes) — it has no field for an arbitrary question, and
extending it would mean changing a contract several earlier tasks
already depend on. `/ask` therefore resolves every question through the
keyword matcher only, in every configuration; no provider is called
from this endpoint today.
