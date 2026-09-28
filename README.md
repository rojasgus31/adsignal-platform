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

The full architecture decision record — why two stores, why the pipeline is
precomputed, why the release gate is AUC and not recall, why `/ask` selects
templates rather than generating SQL, and what the circuit breakers do and do
not protect in the deployed topology — is in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

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

## Deployment

The repository is not published and the service is not deployed yet, so there
are no live links here. The intended topology is three free tiers:

| Piece | Where | Why |
|---|---|---|
| Serving API | Render free web service | 512 MB container; 409-440 MB measured peak across five runs of identical code (see `ARCHITECTURE.md` §8) |
| Serving Postgres | Neon free tier | 0.5 GB, serverless, which is why `warehouse/pool.py` validates connections before handing them out |
| Nightly pipeline | GitHub Actions cron — **planned, not committed** | No always-on daemon is available on a free tier; the job is a once-nightly batch with no latency requirement. No `schedule:` workflow exists yet (see below) |

The Actions runner is ephemeral, so the DuckDB file the pipeline builds
vanishes with the job. `python -m adsignal.cli.load_marts` is the step that
copies the built marts into hosted Postgres, which is what makes the result
outlive the runner.

Set `ADSIGNAL_TRUSTED_PROXY_HOPS=1` behind Render's edge (it terminates TLS
and appends the peer it saw to `X-Forwarded-For`). Leave it at `0` anywhere
the API is reached directly — too high lets a caller forge an address and
evade the rate limiter.

Set `ADSIGNAL_CORS_ORIGINS` to the exact dashboard origin. Wildcards are
rejected by `Settings`, and empty means no browser may call the API at all.

The API sleeps after 15 minutes idle on the free tier, so the first request
after a quiet period waits out a cold start of roughly a minute. The nightly
cron workflow itself is not committed yet; `ci.yml` and `eval.yml` are
triggered by push and pull request.

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
| `GET /campaigns` | Every known campaign, with its campaign and advertiser display names |
| `GET /campaigns/{id}/metrics` | One campaign's full daily time series |
| `GET /anomalies` | One bounded page of current anomaly predictions, each carrying its campaign name (`limit`, `offset`) |
| `GET /anomalies/{campaign_id}/{event_date}` | One campaign-day's prediction |
| `GET /models` | One bounded page of training-run evaluations, newest first (`limit`, `offset`) |
| `GET /benchmark` | Recent runs plus the most recently stored narrative: its prose, date, provider name and generation time (null before the narrative job has run) |
| `POST /search` | Semantic search over stored entity embeddings |
| `POST /score` | Score one campaign-day on demand, with the loaded models |
| `POST /ask` | A natural-language question, answered from precomputed data |

Run it with `docker compose up -d api`, or locally with
`uv run uvicorn adsignal.api.app:create_app --factory --port 8000`. The
router layer issues no SQL of its own: every endpoint reads through a
`warehouse/` reader, and each of those creates its tables if they do not
exist yet, so a deployment where the pipeline has never run answers with
empty results rather than an error — including when several of those
requests arrive at once, which is what the shared advisory lock in
`warehouse/schema.py` is for. Every collection endpoint is bounded: no
response grows with the size of a table.

### Precomputation rule

Every endpoint except `/search`, `/score` and `/ask` reads rows the
pipeline already computed and stored — visitor traffic against them
triggers no inference and no external call, regardless of load. Those
three are the exceptions: `/search` runs an ONNX embedding pass over the
caller's text, `/score` builds one campaign-day's features and runs the
fitted detector and triage models against them, and `/ask` resolves a
template and runs one indexed query. All three are therefore rate-limited
per caller IP, with separate budgets (`/score` tightest, then `/search`,
since their calls cost inference), and `/ask` is additionally cached
(below).

`/score` exists to make the claim falsifiable: `GET /anomalies` returns
what last night's batch decided, and a stored row proves only that a row
was stored. `/score` runs the same two estimators in the API process,
now. It refuses rather than guesses in two cases — a campaign-day with
fewer than seven prior days has NaN rolling baselines and is a 404 naming
the shortfall, and a deployment with no trained artifacts is a 503 —
because a confidently wrong score is worse than a refusal.

### The narrative provider chain

`narrative/build.py` assembles a chain of providers in a fixed order:
Gemini, then Groq, then Jev, then Anthropic — each included only if its
API key is configured — with a `StubProvider` always last. Free tiers
are tried first, the paid provider last, and the stub guarantees a
deployment with no keys at all still renders a narrative (placeholder
text) rather than erroring. Each provider sits behind its own circuit
breaker in `NarrativeChain`; a provider that has failed enough times in
a row is skipped without being called until its cooldown elapses, so
one dead backend costs nothing on later requests. The chain is built
once, as a Dagster resource in `pipeline/definitions.py`, rather than per
materialisation — breaker state rebuilt every run is state that never
fires. The chain also holds a **total budget** across every provider
(`ADSIGNAL_NARRATIVE_BUDGET_SECONDS`, 45 s by default): a per-provider
timeout does not bound the chain, and a provider that answers only after
the budget has run out is recorded as a failure, so a merely-slow backend
can open its breaker too.

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

Because `/ask` is one of the two endpoints that do live work per request,
it is guarded by a per-IP `TokenBucket` (`adsignal/api/ratelimit.py`) — a
token bucket refilled gradually over time, returning 429 once a
caller's budget is exhausted — and identical questions are cached for a
short TTL so a repeated question is answered once, not recomputed on
every call. The rate limiter is checked on every request regardless of
a cache hit, so a burst of one repeated question still trips the
limiter; the cache only saves the downstream template resolution and
database round-trip. Both are per-app collaborators built in
`create_app` and injected through `api/deps.py`, and the cache is
bounded (LRU, hard capacity) as well as TTL'd, because its key is
caller-supplied free text. The limiter's own map of client keys is
bounded the same way, for the same reason.

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

## Deliberate cuts

Things this project knowingly does not do, gathered in one place. Each is
explained where it is implemented and, for the larger ones, in
`ARCHITECTURE.md`.

- **`/ask` does not generate SQL.** A keyword matcher picks among four named
  templates; an unrecognised name is rejected before execution. Structural,
  not sanitised.
- **No LLM is involved in `/ask`'s template selection, in any
  configuration.** `NarrativeProvider.generate(request: NarrativeRequest)`
  takes a closed dataclass of precomputed statistics with no field for free
  text, and that contract predates this endpoint.
- **The release gate is triage macro F1 and detector ROC AUC — not recall.**
  Recall on this detector moves with the dataset's anomaly rate, so it is
  reported but never gated on.
- **The detector is an F1-optimal pre-filter, not a recall-oriented one.**
  Most real anomalies never reach the classifier. A lower-precision
  pre-filter would make the triage output unusable.
- **The end-to-end pipeline metrics are reported but not gated on.**
- **Circuit breaker state does not survive a nightly run.** The chain is
  built at module import, `dagster dev` runs each job in a fresh subprocess,
  so the breakers reset every night and never skip anything in production.
  The chain's total time budget is the real protection.
- **The nightly cron workflow is not committed yet.** The Dagster
  `ScheduleDefinition` and the `load_marts` entry point exist; the Actions
  workflow that would drive them does not.
- **Connection pooling is per-process only.** Two workers do not share
  connections and their pool budgets add up against the database's ceiling.
- **No authentication and no cookies.** The API is read-only and public;
  `allow_credentials` is off and origins are named exactly.
- **No keep-alive ping.** The free tier sleeps after 15 minutes idle and the
  first visitor pays the cold start.
- **`stg_creatives` and the advertiser dimension have no consumer here.**
  They exist for the dashboard's planned vertical filtering.
- **Drift monitoring is forty hand-written lines, not a library.** PSI and a
  two-sample KS test, so every number is explainable.
- **`scripts/measure_memory.py` is only correct inside the container.**
  `ru_maxrss` is kilobytes on Linux and bytes on macOS, and the script divides
  by 1024 either way; run on a macOS host it reports a figure 1024x too
  **large** (a 412 MB process reads as 421,888 "MB"). It also misses the container's
  interpreter, onnxruntime arenas and loaded model.
- **Nothing runs that script automatically.** The 512 MB headroom is
  checkable on demand, not enforced by CI.
