"""Persisting generated narratives to Postgres for the serving layer to read.

Narratives are an append-only history, like `model_metrics`: the dashboard
plots them over time, so a new run never overwrites an older one. A same-day
regeneration is a second row, not a replacement — `read_latest_narrative`
orders by `as_of DESC, id DESC` so the newer row wins.
"""

from dataclasses import dataclass
from datetime import date, datetime

import psycopg

from adsignal.narrative.types import NarrativeResult

_CREATE = """
CREATE TABLE IF NOT EXISTS narrative_runs (
    id           BIGSERIAL PRIMARY KEY,
    as_of        DATE        NOT NULL,
    text         TEXT        NOT NULL,
    provider     TEXT        NOT NULL,
    latency_ms   DOUBLE PRECISION NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_narrative_runs_time ON narrative_runs (as_of DESC, id DESC);
"""


@dataclass(frozen=True, slots=True)
class StoredNarrative:
    """A narrative row as read back from `narrative_runs`."""

    as_of: date
    text: str
    provider: str
    latency_ms: float
    generated_at: datetime


def ensure_narrative_schema(dsn: str) -> None:
    """Create the narrative table if it does not exist."""
    with psycopg.connect(dsn) as conn:
        conn.execute(_CREATE)
        conn.commit()


def write_narrative(dsn: str, as_of: date, result: NarrativeResult) -> None:
    """Append one generated narrative. History is never overwritten."""
    ensure_narrative_schema(dsn)
    with psycopg.connect(dsn) as conn:
        conn.execute(
            """
            INSERT INTO narrative_runs (as_of, text, provider, latency_ms)
            VALUES (%s, %s, %s, %s)
            """,
            (as_of, result.text, result.provider, result.latency_ms),
        )
        conn.commit()


def read_latest_narrative(dsn: str) -> StoredNarrative | None:
    """Return the most recent narrative, or None if the table is empty."""
    ensure_narrative_schema(dsn)
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT as_of, text, provider, latency_ms, generated_at FROM narrative_runs "
            "ORDER BY as_of DESC, id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return StoredNarrative(
        as_of=row[0],
        text=row[1],
        provider=row[2],
        latency_ms=row[3],
        generated_at=row[4],
    )
