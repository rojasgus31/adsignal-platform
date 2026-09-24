"""Persisting model outputs to Postgres for the serving layer to read.

Metrics and drift are append-only histories: the dashboard plots them over
time, so a run never overwrites an earlier one. Predictions are replaced per
run, since a re-run of the same run id is a correction, not a second opinion.
"""

import json
from datetime import datetime

import pandas as pd
import psycopg

from adsignal.ml.drift import FeatureDrift
from adsignal.ml.evaluation import DetectorMetrics, TriageMetrics

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
    per_class_f1       JSONB       NOT NULL,
    confusion          JSONB       NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_metrics_time ON model_metrics (trained_at DESC);

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


def ensure_ml_schema(dsn: str) -> None:
    """Create the ML output tables if they do not exist."""
    with psycopg.connect(dsn) as conn:
        conn.execute(_CREATE_TABLES)
        conn.commit()


def write_model_metrics(
    dsn: str,
    run_id: str,
    trained_at: datetime,
    triage: TriageMetrics,
    detector: DetectorMetrics,
) -> None:
    """Append one evaluation result. History is never overwritten."""
    ensure_ml_schema(dsn)
    with psycopg.connect(dsn) as conn:
        conn.execute(
            """
            INSERT INTO model_metrics (
                run_id, trained_at, triage_accuracy, triage_macro_f1, triage_n_samples,
                detector_precision, detector_recall, detector_f1, per_class_f1, confusion
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
    with psycopg.connect(dsn) as conn:
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
    """Upsert predictions keyed on (event_date, campaign_id). Returns the count."""
    ensure_ml_schema(dsn)
    if predictions.empty:
        return 0

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

    with psycopg.connect(dsn) as conn:
        conn.cursor().executemany(
            """
            INSERT INTO anomaly_predictions
                (event_date, campaign_id, run_id, cause, confidence, severity,
                 escalate, anomaly_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (event_date, campaign_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                cause = EXCLUDED.cause,
                confidence = EXCLUDED.confidence,
                severity = EXCLUDED.severity,
                escalate = EXCLUDED.escalate,
                anomaly_score = EXCLUDED.anomaly_score
            """,
            rows,
        )
        conn.commit()

    return len(rows)
