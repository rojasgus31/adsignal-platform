"""Reading model evaluation history and anomaly predictions back out of Postgres.

`metrics_store.py` only writes `model_metrics`, `drift_reports` and
`anomaly_predictions` — nothing in that module reads them back, and its
writers are pinned signatures for this task, not to be touched. This module
is the api layer's one dependency for those two tables, so the layering rule
("api reads through warehouse/, never issues SQL itself") holds without
modifying `metrics_store.py`.

This module used to also expose `read_latest_artifact_dir`, resolving
`model_metrics.artifact_uri` to a filesystem directory for `api.model_loader`
to load models from. Retired (not kept): the API and the nightly training job
do not share a filesystem in production, so that directory was only ever
valid on the pipeline's own container, and `api.model_loader` now reads
pickled model bytes from Postgres instead - see
`adsignal.warehouse.artifact_store`.
"""

from dataclasses import dataclass
from datetime import date, datetime

from adsignal.warehouse.loader import ensure_schema as ensure_mart_schema
from adsignal.warehouse.metrics_store import ensure_ml_schema
from adsignal.warehouse.paging import DEFAULT_SERVING_PAGE_LIMIT
from adsignal.warehouse.pool import connection

# Every anomaly row carries a display name. `anomaly_predictions` stores a
# campaign_id and nothing else about the campaign, so the name is joined from
# `dim_campaign` in the same query rather than looked up once per row by the
# caller. The LEFT JOIN falls back to the identifier because nothing in the
# schema forces a predicted campaign to have a dimension row, and an INNER
# JOIN would drop those predictions from the response entirely.
_ANOMALY_SELECT = """
SELECT predictions.campaign_id,
       COALESCE(dim.campaign_name, predictions.campaign_id) AS campaign_name,
       predictions.event_date, predictions.run_id, predictions.cause,
       predictions.confidence, predictions.severity, predictions.escalate,
       predictions.anomaly_score
FROM anomaly_predictions AS predictions
LEFT JOIN dim_campaign AS dim ON dim.campaign_id = predictions.campaign_id
"""

_ANOMALY_PAGE_SQL = (
    _ANOMALY_SELECT
    + """
ORDER BY predictions.event_date DESC, predictions.campaign_id
LIMIT %s OFFSET %s
"""
)

_ONE_ANOMALY_SQL = (
    _ANOMALY_SELECT
    + """
WHERE predictions.campaign_id = %s AND predictions.event_date = %s
"""
)


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
class ModelEvidence:
    """Everything `model_metrics` stores about one training run."""

    run_id: str
    trained_at: datetime
    holdout_samples: int
    per_class_f1: dict[str, float]
    confusion: dict[str, dict[str, int]]
    detector_roc_auc: float
    detector_f1: float
    triage_macro_f1: float


@dataclass(frozen=True, slots=True)
class AnomalyPredictionRow:
    """One `anomaly_predictions` row, as read back for the API.

    `campaign_name` is not stored on the row: it is joined from
    `dim_campaign`, and holds `campaign_id` when the dimension has no row for
    the campaign. It is never empty and never absent.
    """

    campaign_id: str
    campaign_name: str
    event_date: date
    run_id: str
    cause: str
    confidence: float
    severity: int
    escalate: bool
    anomaly_score: float


def read_model_metrics(
    dsn: str, limit: int = DEFAULT_SERVING_PAGE_LIMIT, offset: int = 0
) -> list[ModelMetricsRow]:
    """Return one page of training-run evaluations, newest first.

    Bounded in SQL, not in Python: the history grows with every nightly run,
    and an unbounded read serialises all of it into one response body.

    `ensure_ml_schema` runs first so a database that has never had a pipeline
    run against it (the table does not exist yet) answers with an empty list
    rather than a database error.
    """
    ensure_ml_schema(dsn)
    with connection(dsn) as conn:
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


def read_model_evidence(dsn: str, run_id: str) -> ModelEvidence | None:
    """Return everything stored about one training run, or None if it is unknown.

    `run_id` reaches this query bound as a parameter, never interpolated into
    SQL, so a path segment crafted to look like SQL is just a string that
    matches no row - see `test_a_run_id_that_looks_like_sql_is_rejected_safely`.
    """
    ensure_ml_schema(dsn)
    with connection(dsn) as conn:
        row = conn.execute(
            """
            SELECT run_id, trained_at, triage_n_samples, per_class_f1, confusion,
                   detector_roc_auc, detector_f1, triage_macro_f1
            FROM model_metrics
            WHERE run_id = %s
            ORDER BY trained_at DESC, id DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    return ModelEvidence(*row) if row is not None else None


def read_anomaly_predictions(
    dsn: str, limit: int = DEFAULT_SERVING_PAGE_LIMIT, offset: int = 0
) -> list[AnomalyPredictionRow]:
    """Return one page of current anomaly predictions, most recent day first.

    Bounded for the same reason as `read_model_metrics`: every campaign-day
    the current model flags is a row here.

    Both `ensure_*_schema` calls run first because the query spans two
    domains' tables - `anomaly_predictions` from the ML schema and
    `dim_campaign` from the mart schema - so a database missing either one
    answers with an empty list rather than "relation does not exist".
    """
    ensure_ml_schema(dsn)
    ensure_mart_schema(dsn)
    with connection(dsn) as conn:
        rows = conn.execute(_ANOMALY_PAGE_SQL, (limit, offset)).fetchall()
    return [AnomalyPredictionRow(*row) for row in rows]


def read_anomaly_prediction(
    dsn: str, campaign_id: str, event_date: date
) -> AnomalyPredictionRow | None:
    """Return the prediction for one campaign-day, or None if there is none.

    Carries the same joined `campaign_name`, and the same identifier
    fallback, as `read_anomaly_predictions`.
    """
    ensure_ml_schema(dsn)
    ensure_mart_schema(dsn)
    with connection(dsn) as conn:
        row = conn.execute(_ONE_ANOMALY_SQL, (campaign_id, event_date)).fetchone()
    return AnomalyPredictionRow(*row) if row is not None else None
