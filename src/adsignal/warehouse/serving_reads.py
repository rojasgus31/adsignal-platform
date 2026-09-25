"""Reading model evaluation history and anomaly predictions back out of Postgres.

`metrics_store.py` only writes `model_metrics`, `drift_reports` and
`anomaly_predictions` — nothing in that module reads them back, and its
writers are pinned signatures for this task, not to be touched. This module
is the api layer's one dependency for those two tables, so the layering rule
("api reads through warehouse/, never issues SQL itself") holds without
modifying `metrics_store.py`.
"""

from dataclasses import dataclass
from datetime import date, datetime

import psycopg

from adsignal.warehouse.metrics_store import ensure_ml_schema

# The bound both collection readers share, and the ceiling a caller may ask
# for. An unbounded collection is a denial-of-service lever: one response
# otherwise grows with the table, and clients have no pagination contract to
# code against. Defined here, in the layer that enforces them, rather than in
# a router - `api/` depends on `warehouse/`, never the reverse.
DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 500


@dataclass(frozen=True, slots=True)
class ModelMetricsRow:
    """One `model_metrics` row, as read back for the API."""

    run_id: str
    trained_at: datetime
    triage_accuracy: float
    triage_macro_f1: float
    detector_precision: float
    detector_recall: float
    detector_f1: float
    detector_roc_auc: float


@dataclass(frozen=True, slots=True)
class AnomalyPredictionRow:
    """One `anomaly_predictions` row, as read back for the API."""

    campaign_id: str
    event_date: date
    run_id: str
    cause: str
    confidence: float
    severity: int
    escalate: bool
    anomaly_score: float


def read_model_metrics(
    dsn: str, limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0
) -> list[ModelMetricsRow]:
    """Return one page of training-run evaluations, newest first.

    Bounded in SQL, not in Python: the history grows with every nightly run,
    and an unbounded read serialises all of it into one response body.

    `ensure_ml_schema` runs first so a database that has never had a pipeline
    run against it (the table does not exist yet) answers with an empty list
    rather than a database error.
    """
    ensure_ml_schema(dsn)
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            """
            SELECT run_id, trained_at, triage_accuracy, triage_macro_f1,
                   detector_precision, detector_recall, detector_f1, detector_roc_auc
            FROM model_metrics
            ORDER BY trained_at DESC, id DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        ).fetchall()
    return [ModelMetricsRow(*row) for row in rows]


def read_anomaly_predictions(
    dsn: str, limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0
) -> list[AnomalyPredictionRow]:
    """Return one page of current anomaly predictions, most recent day first.

    Bounded for the same reason as `read_model_metrics`: every campaign-day
    the current model flags is a row here.
    """
    ensure_ml_schema(dsn)
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            """
            SELECT campaign_id, event_date, run_id, cause, confidence, severity,
                   escalate, anomaly_score
            FROM anomaly_predictions
            ORDER BY event_date DESC, campaign_id
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        ).fetchall()
    return [AnomalyPredictionRow(*row) for row in rows]


def read_anomaly_prediction(
    dsn: str, campaign_id: str, event_date: date
) -> AnomalyPredictionRow | None:
    """Return the prediction for one campaign-day, or None if there is none."""
    ensure_ml_schema(dsn)
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """
            SELECT campaign_id, event_date, run_id, cause, confidence, severity,
                   escalate, anomaly_score
            FROM anomaly_predictions
            WHERE campaign_id = %s AND event_date = %s
            """,
            (campaign_id, event_date),
        ).fetchone()
    return AnomalyPredictionRow(*row) if row is not None else None
