"""The pipeline's own operations log.

Dagster keeps run history in its own storage. `dagster dev` on an ephemeral
GitHub Actions runner discards that storage when the job ends, so nothing
about a nightly run survives the runner it ran on. This table is a deliberate
duplication of that Dagster feature: it is written by the assets themselves
(see `adsignal.pipeline.assets._record`), so a deployed dashboard can show
what the pipeline did on a machine that no longer exists.

`detail` is stored verbatim, with no scrubbing on the way in. This row is
served by a public endpoint later in this plan, so callers MUST pass only an
exception's TYPE name (`type(error).__name__`), never `str(error)` or any
other message - a message can carry a DSN, and a DSN can carry a password.
This module cannot enforce that contract (it has no way to distinguish a safe
type name from an unsafe message); it is the caller's responsibility, and the
one caller in this codebase (`adsignal.pipeline.assets._record`) honours it.
"""

from dataclasses import dataclass
from datetime import datetime

from adsignal.warehouse.pool import connection
from adsignal.warehouse.schema import apply_schema

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id               BIGSERIAL PRIMARY KEY,
    run_id           TEXT NOT NULL,
    asset            TEXT NOT NULL,
    started_at       TIMESTAMPTZ NOT NULL,
    duration_seconds DOUBLE PRECISION NOT NULL,
    rows_written     INTEGER NOT NULL,
    status           TEXT NOT NULL,
    detail           TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_time ON pipeline_runs (started_at DESC, id DESC);
"""

_INSERT = """
INSERT INTO pipeline_runs
    (run_id, asset, started_at, duration_seconds, rows_written, status, detail)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

_SELECT = """
SELECT run_id, asset, started_at, duration_seconds, rows_written, status, detail
FROM pipeline_runs
ORDER BY started_at DESC, id DESC
LIMIT %s OFFSET %s
"""


@dataclass(frozen=True, slots=True)
class RunEntry:
    """One asset's execution within one pipeline run."""

    run_id: str
    asset: str
    started_at: datetime
    duration_seconds: float
    rows_written: int
    status: str
    detail: str


def ensure_run_log_schema(dsn: str) -> None:
    """Create `pipeline_runs` if it does not exist.

    Serialised against every other `ensure_*_schema` helper by
    `apply_schema`; see `warehouse.schema` for why concurrent `IF NOT EXISTS`
    DDL needs a lock.
    """
    apply_schema(dsn, _CREATE_TABLE)


def record_run(dsn: str, entry: RunEntry) -> None:
    """Append one row. The log is append-only; a run is never edited or deleted."""
    ensure_run_log_schema(dsn)
    with connection(dsn) as conn:
        conn.execute(
            _INSERT,
            (
                entry.run_id,
                entry.asset,
                entry.started_at,
                entry.duration_seconds,
                entry.rows_written,
                entry.status,
                entry.detail,
            ),
        )
        conn.commit()


def read_runs(dsn: str, limit: int = 50, offset: int = 0) -> list[RunEntry]:
    """Return one page of runs, newest first.

    Bounded in SQL, like `warehouse.serving_reads`: the log grows with every
    asset materialization, and an unbounded read would serialise all of it
    into one response body.

    `ensure_run_log_schema` runs first so a database that has never had the
    pipeline run against it (the table does not exist yet) answers with an
    empty list rather than a database error.
    """
    ensure_run_log_schema(dsn)
    with connection(dsn) as conn:
        rows = conn.execute(_SELECT, (limit, offset)).fetchall()
    return [RunEntry(*row) for row in rows]
