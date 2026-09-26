"""Storing and reading trained model artifacts in Postgres.

`ml.registry.save_models`/`load_models` persist a training run's fitted
models to a filesystem directory. That works as long as the training job and
the serving API share a filesystem; on a nightly-cron deployment where they
run as separate processes with separate disks, a directory path is only ever
valid on the machine that wrote it, and `POST /score` degrades to a
permanent 503. Postgres is the one thing both processes already share, so
this module makes it the artifact store the serving path actually reads.

One row per training run, keyed by `run_id` - the same identity
`model_metrics` already uses for a run, so the two tables describe the same
run without a second, independent numbering scheme. `sklearn_version` and
`python_version` are recorded because a pickled estimator is only loadable by
a compatible scikit-learn: without them, a serving image that has drifted
from the training image gets a bare, unreadable traceback instead of a
refusal naming what actually differs. See `api.model_loader` for how the
versions are used.
"""

from dataclasses import dataclass

from adsignal.warehouse.pool import connection
from adsignal.warehouse.schema import apply_schema

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS model_artifacts (
    run_id          TEXT        PRIMARY KEY,
    detector_bytes  BYTEA       NOT NULL,
    triage_bytes    BYTEA       NOT NULL,
    sklearn_version TEXT        NOT NULL,
    python_version  TEXT        NOT NULL,
    written_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_time ON model_artifacts (written_at DESC);
"""


@dataclass(frozen=True, slots=True)
class StoredArtifacts:
    """One `model_artifacts` row, as read back for the serving loader."""

    run_id: str
    detector_bytes: bytes
    triage_bytes: bytes
    sklearn_version: str
    python_version: str


def ensure_artifact_schema(dsn: str) -> None:
    """Create the `model_artifacts` table if it does not exist.

    Serialised against every other `ensure_*_schema` helper by
    `apply_schema` - see `warehouse.schema` for why concurrent `IF NOT
    EXISTS` DDL needs a lock. Owned here, not added to
    `warehouse.schema`, which holds only the shared advisory-lock machinery:
    every domain module owns its own DDL.
    """
    apply_schema(dsn, _CREATE_TABLE)


def write_artifacts(
    dsn: str,
    run_id: str,
    detector_bytes: bytes,
    triage_bytes: bytes,
    sklearn_version: str,
    python_version: str,
) -> None:
    """Persist one run's pickled models, replacing any earlier row for `run_id`.

    Pickling happens in the caller (`ml.registry.dumps_models`); this module
    never imports `pickle` and stores exactly the bytes it is handed. A
    second write for the same `run_id` (a retried training step) replaces
    the row rather than erroring or duplicating it - unlike `model_metrics`,
    which is an append-only history, one run has exactly one artifact set.
    """
    ensure_artifact_schema(dsn)
    with connection(dsn) as conn:
        conn.execute(
            """
            INSERT INTO model_artifacts (
                run_id, detector_bytes, triage_bytes, sklearn_version, python_version
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                detector_bytes = EXCLUDED.detector_bytes,
                triage_bytes = EXCLUDED.triage_bytes,
                sklearn_version = EXCLUDED.sklearn_version,
                python_version = EXCLUDED.python_version,
                written_at = now()
            """,
            (run_id, detector_bytes, triage_bytes, sklearn_version, python_version),
        )
        conn.commit()


def read_latest_artifacts(dsn: str) -> StoredArtifacts | None:
    """Return the newest training run's stored artifacts, or None if none exist.

    "Newest" by this table's own `written_at`, not by `model_metrics.
    trained_at` - this module never reads `model_metrics`, the same
    independence `narrative_store` keeps from the mart it is not about. A
    caller that needs the two to agree (the pipeline, writing both for one
    run) is responsible for that; a reader here just wants the most recently
    stored artifact set.
    """
    ensure_artifact_schema(dsn)
    with connection(dsn) as conn:
        row = conn.execute(
            """
            SELECT run_id, detector_bytes, triage_bytes, sklearn_version, python_version
            FROM model_artifacts
            ORDER BY written_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    run_id, detector_bytes, triage_bytes, sklearn_version, python_version = row
    return StoredArtifacts(
        run_id=str(run_id),
        detector_bytes=bytes(detector_bytes),
        triage_bytes=bytes(triage_bytes),
        sklearn_version=str(sklearn_version),
        python_version=str(python_version),
    )
