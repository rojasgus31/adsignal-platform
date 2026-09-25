"""Persisting model outputs to Postgres for the serving layer to read.

Metrics and drift are append-only histories: the dashboard plots them over
time, so a run never overwrites an earlier one. Predictions are a full
refresh, like `warehouse.loader`'s mart load: `anomaly_scores` always scores
the entire mart, never a date-limited slice, so a campaign-day a previous run
flagged and this run does not is stale, not a second opinion, and must not
outlive the model that produced it (retention policy: see
`write_predictions`).
"""

import json
from datetime import datetime

import pandas as pd

from adsignal.ml.drift import FeatureDrift
from adsignal.ml.evaluation import DetectorMetrics, TriageMetrics
from adsignal.warehouse.pool import connection
from adsignal.warehouse.schema import apply_schema

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS model_metrics (
    id                 BIGSERIAL PRIMARY KEY,
    run_id             TEXT        NOT NULL,
    trained_at         TIMESTAMPTZ NOT NULL,
    triage_accuracy    DOUBLE PRECISION NOT NULL,
    triage_macro_f1    DOUBLE PRECISION NOT NULL,
    triage_n_samples   INTEGER     NOT NULL,
    detector_precision DOUBLE PRECISION NOT NULL,
    detector_recall    DOUBLE PRECISION NOT NULL,
    detector_f1        DOUBLE PRECISION NOT NULL,
    detector_roc_auc   DOUBLE PRECISION NOT NULL,
    artifact_uri       TEXT        NOT NULL,
    per_class_f1       JSONB       NOT NULL,
    confusion          JSONB       NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_metrics_time ON model_metrics (trained_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS drift_reports (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT        NOT NULL,
    measured_at   TIMESTAMPTZ NOT NULL,
    feature       TEXT        NOT NULL,
    psi           DOUBLE PRECISION NOT NULL,
    ks_statistic  DOUBLE PRECISION NOT NULL,
    ks_pvalue     DOUBLE PRECISION NOT NULL,
    drifted       BOOLEAN     NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_drift_time ON drift_reports (measured_at DESC);

CREATE TABLE IF NOT EXISTS anomaly_predictions (
    event_date    DATE        NOT NULL,
    campaign_id   TEXT        NOT NULL,
    run_id        TEXT        NOT NULL,
    cause         TEXT        NOT NULL,
    confidence    DOUBLE PRECISION NOT NULL,
    severity      INTEGER     NOT NULL,
    escalate      BOOLEAN     NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (event_date, campaign_id)
);
CREATE INDEX IF NOT EXISTS idx_predictions_escalate
    ON anomaly_predictions (event_date DESC) WHERE escalate;
"""

# `CREATE TABLE IF NOT EXISTS` above only covers a fresh database; a database
# created before `detector_roc_auc`/`artifact_uri` existed needs these added
# explicitly. Nullable here (existing rows have neither) even though a fresh
# table declares them NOT NULL - `write_model_metrics` always supplies both.
_MIGRATE_MODEL_METRICS = """
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS detector_roc_auc DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS artifact_uri TEXT;
"""


def ensure_ml_schema(dsn: str) -> None:
    """Create the ML output tables if they do not exist, and migrate old ones.

    Serialised against every other `ensure_*_schema` helper by
    `apply_schema`; see that module for why concurrent `IF NOT EXISTS` DDL
    needs a lock.
    """
    apply_schema(dsn, _CREATE_TABLES, _MIGRATE_MODEL_METRICS)


def write_model_metrics(
    dsn: str,
    run_id: str,
    trained_at: datetime,
    triage: TriageMetrics,
    detector: DetectorMetrics,
    artifact_uri: str,
) -> None:
    """Append one evaluation result. History is never overwritten.

    `artifact_uri` is the directory `registry.log_training_run` wrote this
    run's models to, recorded alongside the run id it belongs to so a reader
    (`adsignal.pipeline.assets._latest_run`) never has to guess where a run's
    artifacts live from a separate, per-process `Settings.mlflow_dir` (I4).
    """
    ensure_ml_schema(dsn)
    with connection(dsn) as conn:
        conn.execute(
            """
            INSERT INTO model_metrics (
                run_id, trained_at, triage_accuracy, triage_macro_f1, triage_n_samples,
                detector_precision, detector_recall, detector_f1, detector_roc_auc,
                artifact_uri, per_class_f1, confusion
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id,
                trained_at,
                triage.accuracy,
                triage.macro_f1,
                triage.n_samples,
                detector.precision,
                detector.recall,
                detector.f1,
                detector.roc_auc,
                artifact_uri,
                json.dumps(triage.per_class_f1),
                json.dumps(triage.confusion),
            ),
        )
        conn.commit()


def write_drift_report(
    dsn: str,
    run_id: str,
    measured_at: datetime,
    reports: list[FeatureDrift],
) -> None:
    """Append one drift measurement per feature."""
    ensure_ml_schema(dsn)
    if not reports:
        return
    with connection(dsn) as conn:
        conn.cursor().executemany(
            """
            INSERT INTO drift_reports
                (run_id, measured_at, feature, psi, ks_statistic, ks_pvalue, drifted)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            [
                (
                    run_id,
                    measured_at,
                    report.feature,
                    report.psi,
                    report.ks_statistic,
                    report.ks_pvalue,
                    report.drifted,
                )
                for report in reports
            ],
        )
        conn.commit()


def write_predictions(dsn: str, run_id: str, predictions: pd.DataFrame) -> int:
    """Replace every prediction with this run's. Full refresh. Returns the count.

    `anomaly_scores` always scores the entire mart, so the previous run's
    complete set of predictions is exactly as stale as the model that
    produced them - a per-key upsert would leave a campaign-day the current
    model no longer flags sitting in the table forever, attributed to a
    retired run (I9). Delete-then-insert in one transaction, the same choice
    `warehouse.loader` makes for the mart itself and for the same reason.
    """
    ensure_ml_schema(dsn)

    event_dates = pd.to_datetime(predictions["event_date"]).dt.date.tolist()
    campaign_ids = predictions["campaign_id"].astype(str).tolist()
    causes = predictions["cause"].astype(str).tolist()
    confidences = predictions["confidence"].astype(float).tolist()
    severities = predictions["severity"].astype(int).tolist()
    escalates = predictions["escalate"].astype(bool).tolist()
    anomaly_scores = predictions["anomaly_score"].astype(float).tolist()

    rows = [
        (
            event_dates[i],
            campaign_ids[i],
            run_id,
            causes[i],
            confidences[i],
            severities[i],
            escalates[i],
            anomaly_scores[i],
        )
        for i in range(len(predictions))
    ]

    with connection(dsn) as conn:
        conn.execute("DELETE FROM anomaly_predictions")
        if rows:
            conn.cursor().executemany(
                """
                INSERT INTO anomaly_predictions
                    (event_date, campaign_id, run_id, cause, confidence, severity,
                     escalate, anomaly_score)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                rows,
            )
        conn.commit()

    return len(rows)
