import os
from datetime import UTC, datetime

import pandas as pd
import psycopg
import pytest

from adsignal.ml.drift import FeatureDrift
from adsignal.ml.evaluation import DetectorMetrics, TriageMetrics
from adsignal.warehouse.metrics_store import (
    ensure_ml_schema,
    write_drift_report,
    write_model_metrics,
    write_predictions,
)

DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


def _postgres_available() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _postgres_available(), reason="Postgres not reachable")


@pytest.fixture(autouse=True)
def clean_tables():
    ensure_ml_schema(DSN)
    with psycopg.connect(DSN) as conn:
        for table in ("model_metrics", "drift_reports", "anomaly_predictions"):
            conn.execute(f"TRUNCATE TABLE {table}")  # noqa: S608
        conn.commit()
    yield


def _triage_metrics() -> TriageMetrics:
    return TriageMetrics(
        accuracy=0.8,
        macro_f1=0.75,
        per_class_f1={"budget_pacing": 0.8, "tracking_error": 0.7},
        confusion={"budget_pacing": {"budget_pacing": 4, "tracking_error": 1}},
        n_samples=5,
    )


def _detector_metrics() -> DetectorMetrics:
    return DetectorMetrics(
        precision=0.6, recall=0.9, f1=0.72, roc_auc=0.85, n_samples=100, n_flagged=15
    )


def test_model_metrics_round_trip() -> None:
    write_model_metrics(
        DSN, "run_1", datetime.now(UTC), _triage_metrics(), _detector_metrics(), "/mlruns/run_1"
    )
    with psycopg.connect(DSN) as conn:
        row = conn.execute(
            "SELECT run_id, triage_macro_f1, detector_recall FROM model_metrics"
        ).fetchone()
    assert row[0] == "run_1"
    assert row[1] == pytest.approx(0.75)
    assert row[2] == pytest.approx(0.9)


def test_model_metrics_records_the_gated_detector_metric_and_its_artifact_location() -> None:
    """I8/I4: the AUC the gate actually tests, and where the run's models live, are both stored."""
    write_model_metrics(
        DSN, "run_1", datetime.now(UTC), _triage_metrics(), _detector_metrics(), "/mlruns/run_1"
    )
    with psycopg.connect(DSN) as conn:
        row = conn.execute("SELECT detector_roc_auc, artifact_uri FROM model_metrics").fetchone()
    assert row[0] == pytest.approx(0.85)
    assert row[1] == "/mlruns/run_1"


def test_model_metrics_keep_history_rather_than_overwriting() -> None:
    write_model_metrics(
        DSN, "run_1", datetime.now(UTC), _triage_metrics(), _detector_metrics(), "/mlruns/run_1"
    )
    write_model_metrics(
        DSN, "run_2", datetime.now(UTC), _triage_metrics(), _detector_metrics(), "/mlruns/run_2"
    )
    with psycopg.connect(DSN) as conn:
        count = conn.execute("SELECT count(*) FROM model_metrics").fetchone()[0]
    assert count == 2, "metrics history was overwritten; the dashboard needs a series"


def test_drift_report_round_trip() -> None:
    reports = [
        FeatureDrift("spend_ratio", 0.31, 0.22, 0.001, True),
        FeatureDrift("ctr_ratio", 0.04, 0.03, 0.8, False),
    ]
    write_drift_report(DSN, "run_1", datetime.now(UTC), reports)
    with psycopg.connect(DSN) as conn:
        rows = conn.execute(
            "SELECT feature, psi, drifted FROM drift_reports ORDER BY feature"
        ).fetchall()
    assert len(rows) == 2
    assert rows[1][0] == "spend_ratio"
    assert rows[1][2] is True


def test_predictions_round_trip_and_are_replaced_per_run() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "campaign_id": ["cmp_a", "cmp_a"],
            "cause": ["budget_pacing", "tracking_error"],
            "confidence": [0.9, 0.7],
            "severity": [4, 2],
            "escalate": [True, False],
            "anomaly_score": [0.8, 0.3],
        }
    )
    assert write_predictions(DSN, "run_1", frame) == 2
    assert write_predictions(DSN, "run_1", frame) == 2

    with psycopg.connect(DSN) as conn:
        count = conn.execute("SELECT count(*) FROM anomaly_predictions").fetchone()[0]
    assert count == 2, "re-running the same run duplicated predictions"


def test_predictions_from_a_retired_run_do_not_outlive_it() -> None:
    """I9: a campaign-day the current run does not flag must not remain in the

    table forever under a retired run's id. `anomaly_scores` always scores
    the whole mart, so it is stale, not a second opinion.
    """
    run_a = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "campaign_id": ["cmp_a", "cmp_b", "cmp_c"],
            "cause": ["budget_pacing", "tracking_error", "creative_fatigue"],
            "confidence": [0.9, 0.7, 0.6],
            "severity": [4, 2, 3],
            "escalate": [True, False, False],
            "anomaly_score": [0.8, 0.3, 0.5],
        }
    )
    run_b = run_a.iloc[[0]]  # the next run only still flags cmp_a

    assert write_predictions(DSN, "run_a", run_a) == 3
    assert write_predictions(DSN, "run_b", run_b) == 1

    with psycopg.connect(DSN) as conn:
        rows = conn.execute("SELECT campaign_id, run_id FROM anomaly_predictions").fetchall()
    assert [(campaign_id, run_id) for campaign_id, run_id in rows] == [("cmp_a", "run_b")]


def test_ensure_ml_schema_is_idempotent() -> None:
    ensure_ml_schema(DSN)
    ensure_ml_schema(DSN)
