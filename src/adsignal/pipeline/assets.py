"""Dagster assets for the AdSignal data platform.

Each asset is idempotent: re-materializing it produces the same end state.
"""

from datetime import UTC, datetime
from pathlib import Path

import psycopg
from dagster import AssetExecutionContext, MetadataValue, asset

from adsignal.config import Settings
from adsignal.generator.anomalies import AnomalyInjector
from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events
from adsignal.ml.detector import DETECTOR_FEATURES, AnomalyDetector
from adsignal.ml.drift import feature_drift
from adsignal.ml.evaluation import evaluate_detector, evaluate_triage
from adsignal.ml.features import build_features
from adsignal.ml.registry import load_models, log_training_run
from adsignal.ml.splits import temporal_split
from adsignal.ml.triage import TriageClassifier
from adsignal.warehouse.duckdb_io import open_duckdb, register_sources, run_dbt
from adsignal.warehouse.loader import load_mart_to_postgres
from adsignal.warehouse.metrics_store import (
    write_drift_report,
    write_model_metrics,
    write_predictions,
)
from adsignal.warehouse.reader import read_campaign_daily
from adsignal.warehouse.writer import write_events_parquet, write_labels_parquet

_DEFAULT_DBT_DIR = Path(__file__).resolve().parents[3] / "dbt"

# campaigns per advertiser x creatives per campaign x mean campaign duration
_ROWS_PER_ADVERTISER = 8 * 4 * 105


def _advertisers_for(target_events: int) -> int:
    """Pick an advertiser count whose expected row volume approximates the target."""
    return max(1, round(target_events / _ROWS_PER_ADVERTISER))


def _dbt_project_dir(settings: Settings) -> Path:
    """Resolve the dbt project directory, honoring a container override.

    `settings.dbt_dir` is set by the ADSIGNAL_DBT_DIR environment variable when
    the dbt project is copied to a fixed location (e.g. inside a container).
    Falls back to the path relative to this source file for a source checkout.
    """
    return settings.dbt_dir if settings.dbt_dir is not None else _DEFAULT_DBT_DIR


@asset(required_resource_keys={"settings"}, description="Synthetic events and ground-truth labels")
def catalog_and_raw_events(context: AssetExecutionContext) -> None:
    """Generate the catalog, stream events to Parquet, and record anomaly labels."""
    settings: Settings = context.resources.settings

    catalog = build_catalog(
        seed=settings.seed,
        start_date=settings.start_date,
        end_date=settings.end_date,
        n_advertisers=_advertisers_for(settings.target_events),
    )
    injector = AnomalyInjector(seed=settings.seed, rate=0.01)

    row_count = write_events_parquet(
        injector.rows(generate_events(catalog, seed=settings.seed)),
        settings.raw_events_path,
    )
    label_count = write_labels_parquet(injector.labels, settings.anomaly_labels_path)

    with open_duckdb(settings.duckdb_path) as conn:
        register_sources(
            conn,
            events_path=settings.raw_events_path,
            labels_path=settings.anomaly_labels_path,
            catalog=catalog,
        )

    context.add_output_metadata(
        {
            "row_count": MetadataValue.int(row_count),
            "label_count": MetadataValue.int(label_count),
            "campaigns": MetadataValue.int(len(catalog.campaigns)),
            "events_path": MetadataValue.path(str(settings.raw_events_path)),
        }
    )


@asset(
    deps=[catalog_and_raw_events],
    required_resource_keys={"settings"},
    description="dbt staging views and the campaign-daily fact table in DuckDB",
)
def dbt_marts(context: AssetExecutionContext) -> None:
    """Run `dbt build`, which materializes the models and runs the data tests."""
    settings: Settings = context.resources.settings
    project_dir = _dbt_project_dir(settings)
    run_dbt(project_dir=project_dir, duckdb_path=settings.duckdb_path, command="build")
    context.add_output_metadata({"dbt_project": MetadataValue.path(str(project_dir))})


@asset(
    deps=[dbt_marts],
    required_resource_keys={"settings"},
    description="Aggregated marts upserted into the Postgres serving layer",
)
def campaign_daily_mart(context: AssetExecutionContext) -> None:
    """Upsert the DuckDB fact table into Postgres."""
    settings: Settings = context.resources.settings
    loaded = load_mart_to_postgres(settings.duckdb_path, settings.postgres_dsn)
    context.add_output_metadata({"rows_upserted": MetadataValue.int(loaded)})


_HOLDOUT_FRACTION = 0.25
_BASELINE_WINDOW = 7


@asset(
    deps=[campaign_daily_mart],
    required_resource_keys={"settings"},
    description="Trained anomaly detector and triage classifier, logged to MLflow",
)
def trained_models(context: AssetExecutionContext) -> None:
    """Train both models on a temporal split of the mart and record the run."""
    settings: Settings = context.resources.settings

    features = build_features(
        read_campaign_daily(settings.postgres_dsn), baseline_window=_BASELINE_WINDOW
    )
    train, holdout = temporal_split(features, holdout_fraction=_HOLDOUT_FRACTION)

    detector = AnomalyDetector(random_state=settings.seed).fit(train)
    classifier = TriageClassifier(random_state=settings.seed).fit(train)

    triage_metrics = evaluate_triage(classifier, holdout)
    detector_metrics = evaluate_detector(detector, holdout)

    run_id = log_training_run(
        tracking_dir=settings.mlflow_dir,
        detector=detector,
        classifier=classifier,
        triage_metrics=triage_metrics,
        detector_metrics=detector_metrics,
        params={
            "baseline_window": _BASELINE_WINDOW,
            "holdout_fraction": _HOLDOUT_FRACTION,
            "seed": settings.seed,
        },
    )
    write_model_metrics(
        settings.postgres_dsn, run_id, datetime.now(UTC), triage_metrics, detector_metrics
    )

    context.add_output_metadata(
        {
            "run_id": MetadataValue.text(run_id),
            "triage_macro_f1": MetadataValue.float(triage_metrics.macro_f1),
            "triage_accuracy": MetadataValue.float(triage_metrics.accuracy),
            "detector_recall": MetadataValue.float(detector_metrics.recall),
            "train_rows": MetadataValue.int(len(train)),
            "holdout_rows": MetadataValue.int(len(holdout)),
        }
    )


@asset(
    deps=[trained_models],
    required_resource_keys={"settings"},
    description="Per-campaign-day anomaly scores and triage predictions in Postgres",
)
def anomaly_scores(context: AssetExecutionContext) -> None:
    """Score every campaign-day and upsert the predictions."""
    settings: Settings = context.resources.settings
    run_id = _latest_run_id(settings)

    features = build_features(
        read_campaign_daily(settings.postgres_dsn), baseline_window=_BASELINE_WINDOW
    )
    detector, classifier = load_models(settings.mlflow_dir / "artifacts" / run_id)

    flagged = features[detector.predict(features)]
    if flagged.empty:
        context.add_output_metadata({"rows_written": MetadataValue.int(0)})
        return

    predictions = classifier.predict(flagged)
    predictions["event_date"] = flagged["event_date"].to_numpy()
    predictions["campaign_id"] = flagged["campaign_id"].to_numpy()
    predictions["anomaly_score"] = detector.score(flagged)

    written = write_predictions(settings.postgres_dsn, run_id, predictions)
    context.add_output_metadata({"rows_written": MetadataValue.int(written)})


@asset(
    deps=[trained_models],
    required_resource_keys={"settings"},
    description="Feature drift between the training window and the most recent window",
)
def drift_report(context: AssetExecutionContext) -> None:
    """Compare the holdout window's feature distributions against the training window."""
    settings: Settings = context.resources.settings
    run_id = _latest_run_id(settings)

    features = build_features(
        read_campaign_daily(settings.postgres_dsn), baseline_window=_BASELINE_WINDOW
    )
    reference, current = temporal_split(features, holdout_fraction=_HOLDOUT_FRACTION)

    reports = feature_drift(reference, current, list(DETECTOR_FEATURES))
    write_drift_report(settings.postgres_dsn, run_id, datetime.now(UTC), reports)

    context.add_output_metadata(
        {
            "features_measured": MetadataValue.int(len(reports)),
            "features_drifted": MetadataValue.int(sum(r.drifted for r in reports)),
        }
    )


def _latest_run_id(settings: Settings) -> str:
    """Return the most recent training run id recorded in Postgres."""
    with psycopg.connect(settings.postgres_dsn) as conn:
        row = conn.execute(
            "SELECT run_id FROM model_metrics ORDER BY trained_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("no training run recorded; materialize trained_models first")
    return str(row[0])
