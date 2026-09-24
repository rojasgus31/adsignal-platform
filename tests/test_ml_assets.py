import os
from datetime import date

import psycopg
import pytest
from dagster import materialize

from adsignal.config import Settings
from adsignal.pipeline.assets import (
    anomaly_scores,
    campaign_daily_mart,
    catalog_and_raw_events,
    dbt_marts,
    drift_report,
    trained_models,
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


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    tmp_path = tmp_path_factory.mktemp("ml_assets")
    return Settings(
        target_events=40_000,
        seed=4242,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ml.duckdb",
        postgres_dsn=DSN,
        mlflow_dir=tmp_path / "mlruns",
    )


@pytest.fixture(scope="module")
def pipeline_run(settings: Settings):
    """Materialize the full pipeline, including scoring and drift, exactly once.

    Every test in this module asserts against this single run's outputs and
    the database rows it wrote, instead of re-running the full pipeline
    (data generation, dbt build, training) per test.
    """
    result = materialize(
        [
            catalog_and_raw_events,
            dbt_marts,
            campaign_daily_mart,
            trained_models,
            anomaly_scores,
            drift_report,
        ],
        resources={"settings": settings},
    )
    assert result.success
    return result, settings


def _run_id(pipeline_run) -> str:
    result, _ = pipeline_run
    metadata = result.asset_materializations_for_node("trained_models")[0].metadata
    return str(metadata["run_id"].value)


def test_training_asset_records_metrics_and_a_run_id(pipeline_run) -> None:
    result, settings = pipeline_run
    metadata = result.asset_materializations_for_node("trained_models")[0].metadata
    assert metadata["run_id"].value
    assert 0.0 <= metadata["triage_macro_f1"].value <= 1.0

    run_id = _run_id(pipeline_run)
    with psycopg.connect(settings.postgres_dsn) as conn:
        count = conn.execute(
            "SELECT count(*) FROM model_metrics WHERE run_id = %s", (run_id,)
        ).fetchone()[0]
    # >= rather than == : this fixture's module scope means a re-run within
    # the same test session (e.g. under -p randomly) could add more than one
    # row for the same run_id if trained_models were re-materialized, but it
    # must never be zero - this scoped WHERE is what makes the assertion mean
    # something, unlike an unscoped `count(*) >= 1` on an append-only table
    # that would pass on residue from any earlier test (P1-2).
    assert count >= 1


def test_training_asset_writes_model_artifacts(pipeline_run) -> None:
    """P1-3: the property that matters is the artifact files `anomaly_scores`
    depends on, not merely that `mlflow_dir` (created on the first line of
    `log_training_run`, before anything is saved) exists."""
    _, settings = pipeline_run
    run_id = _run_id(pipeline_run)
    artifact_dir = settings.mlflow_dir / "artifacts" / run_id
    assert (artifact_dir / "detector.pkl").exists()
    assert (artifact_dir / "triage.pkl").exists()


def test_scoring_asset_writes_predictions(pipeline_run) -> None:
    _, settings = pipeline_run
    run_id = _run_id(pipeline_run)
    with psycopg.connect(settings.postgres_dsn) as conn:
        count = conn.execute(
            "SELECT count(*) FROM anomaly_predictions WHERE run_id = %s", (run_id,)
        ).fetchone()[0]
    assert count > 0


def test_predictions_only_carry_known_causes(pipeline_run) -> None:
    from adsignal.generator.models import ANOMALY_CAUSES

    _, settings = pipeline_run
    with psycopg.connect(settings.postgres_dsn) as conn:
        causes = {row[0] for row in conn.execute("SELECT DISTINCT cause FROM anomaly_predictions")}
    assert causes <= set(ANOMALY_CAUSES)


def test_drift_asset_writes_one_row_per_feature(pipeline_run) -> None:
    _, settings = pipeline_run
    run_id = _run_id(pipeline_run)
    with psycopg.connect(settings.postgres_dsn) as conn:
        count = conn.execute(
            "SELECT count(*) FROM drift_reports WHERE run_id = %s", (run_id,)
        ).fetchone()[0]
    assert count > 0


def test_definitions_expose_the_ml_assets() -> None:
    from adsignal.pipeline.definitions import defs

    keys = {key.to_user_string() for key in defs.resolve_asset_graph().get_all_asset_keys()}
    assert {"trained_models", "anomaly_scores", "drift_report"} <= keys
