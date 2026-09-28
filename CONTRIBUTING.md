# Contributing

## Setup

```bash
uv sync
uv run pre-commit install
uv run dbt deps --project-dir dbt --profiles-dir dbt
```

## Before opening a PR

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

`pre-commit` runs the first two (plus a few housekeeping hooks) automatically
on commit; it does not run `mypy` or the test suite, so run those yourself.

## Known limitations

**The test suite cannot be run concurrently with itself.** Every test DSN in
`tests/conftest.py` (`ADSIGNAL_TEST_DSN`, `ADSIGNAL_EMPTY_TEST_DSN`,
`ADSIGNAL_DROPPED_TEST_DSN`, `ADSIGNAL_CONCURRENCY_TEST_DSN`,
`ADSIGNAL_DIM_TEST_DSN`, `ADSIGNAL_DB_ONLY_TEST_DSN`) resolves to a fixed
database name on a shared Postgres server. A second `pytest` process against
the same server truncates the first run's tables mid-run. This isn't
theoretical: it was hit during development, where three `test_score_endpoint`
tests failed with `IndexError` and looked exactly like a serving-layer
regression until the cause was traced back to a second concurrent run.

This is a stated limitation, not a bug to fix here. The six
`ADSIGNAL_*_TEST_DSN` overrides are the seam a per-worker database suffix
would hang off, for whoever eventually wants `pytest-xdist` or CI job
parallelism. Until then: run the suite alone, never in parallel with itself,
and never in parallel with another process pointed at the same test
databases.

## The dashboard used to show identifiers instead of campaign names

`load_dim_campaign` (in `src/adsignal/warehouse/loader.py`) was written,
tested, and read by the API's campaign and anomaly queries — but no Dagster
asset ever called it. Every real pipeline run left `dim_campaign` empty, so
the dashboard rendered `cmp_0058_006` where a campaign name belonged, and no
number of pipeline runs would have changed that. The test suite concealed it:
`tests/conftest.py` calls `load_dim_campaign` itself to build its fixtures, so
every dimension test passed — genuinely, against real data — while validating
the loader in isolation from the only thing meant to invoke it. It was fixed
in `campaign_daily_mart` (commit `b35217f`), whose regression test asserts
against the database after a materialization and never calls the loader
itself, and which checks that every fact campaign *resolves to* a dimension
row rather than merely that rows exist — a `count > 0` check would have
passed against stale fixture rows and certified the bug.

The general lesson: a fixture that performs production's job makes the test
pass and the product broken.
