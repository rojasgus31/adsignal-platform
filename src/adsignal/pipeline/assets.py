"""Dagster assets for the AdSignal data platform.

Each asset is idempotent: re-materializing it produces the same end state.
"""

import platform
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import sklearn
from dagster import AssetExecutionContext, MetadataValue, asset

from adsignal.config import Settings
from adsignal.generator.anomalies import AnomalyInjector
from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events
from adsignal.ml.detector import DETECTOR_FEATURES, AnomalyDetector
from adsignal.ml.drift import feature_drift
from adsignal.ml.embeddings import EmbeddingModel
from adsignal.ml.features import build_features
from adsignal.ml.registry import dumps_models, load_models, log_training_run
from adsignal.ml.splits import temporal_split
from adsignal.ml.training import BASELINE_WINDOW, HOLDOUT_FRACTION, train_and_evaluate
from adsignal.ml.triage import TriageClassifier
from adsignal.narrative.chain import AllProvidersFailed, NarrativeChain
from adsignal.narrative.types import NarrativeRequest
from adsignal.warehouse.artifact_store import write_artifacts
from adsignal.warehouse.duckdb_io import open_duckdb, register_sources, run_dbt
from adsignal.warehouse.loader import load_dim_campaign, load_mart_to_postgres
from adsignal.warehouse.metrics_store import (
    write_drift_report,
    write_model_metrics,
    write_predictions,
)
from adsignal.warehouse.narrative_store import write_narrative
from adsignal.warehouse.reader import read_campaign_daily
from adsignal.warehouse.run_log import RunEntry, record_run
from adsignal.warehouse.text_source import read_entity_text
from adsignal.warehouse.vector_store import EmbeddingRow, upsert_embeddings
from adsignal.warehouse.writer import write_events_parquet, write_labels_parquet


@dataclass(slots=True)
class _RunCounter:
    """A mutable holder an asset can set `rows` on before returning.

    `_record` needs a value to log after the asset body finishes (or raises),
    so the count has to live somewhere the asset can write to as it runs
    rather than as a return value the wrapper never sees.
    """

    rows: int = field(default=0)


@contextmanager
def _record(context: AssetExecutionContext, settings: Settings) -> Iterator[_RunCounter]:
    """Record this asset's execution to `pipeline_runs`, whether it succeeds or raises.

    Dagster's own run history lives in storage `dagster dev` discards when an
    ephemeral CI runner's job ends; this table is written by the asset itself
    so it survives that runner (see `adsignal.warehouse.run_log`).

    `detail` gets the exception's TYPE name only, never `str(error)`: a
    message can carry a DSN, and this row is served by a public endpoint
    later in this plan.

    Catches `BaseException`, not `Exception`: a run killed on an ephemeral CI
    runner (SIGTERM, a Dagster-level cancellation) unwinds as a direct
    `BaseException` subclass, not an `Exception`, and that cancellation is
    exactly the case this table exists to record. Letting it fall through to
    `except Exception` would write a `"succeeded"` row for a run that was
    killed - a lie that is worse than a missing row.

    The `record_run` call itself is guarded: a logging failure (the serving
    Postgres being unreachable) must never mask the asset's real exception by
    replacing it in the `finally`, and must never fail an otherwise-successful
    asset. The log degrades; the pipeline does not.
    """
    counter = _RunCounter()
    started = datetime.now(UTC)
    clock = time.monotonic()
    status, detail = "succeeded", ""
    try:
        yield counter
    except BaseException as error:  # not Exception - see docstring
        status, detail = "failed", type(error).__name__
        raise
    finally:
        try:
            record_run(
                settings.postgres_dsn,
                RunEntry(
                    run_id=context.run_id,
                    asset=context.asset_key.to_user_string(),
                    started_at=started,
                    duration_seconds=time.monotonic() - clock,
                    rows_written=counter.rows,
                    status=status,
                    detail=detail,
                ),
            )
        except Exception as log_error:  # noqa: BLE001 - logging must never break the asset
            context.log.warning(f"could not record run: {type(log_error).__name__}")


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

    with _record(context, settings) as counter:
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

        counter.rows = row_count + label_count
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
    with _record(context, settings) as counter:
        project_dir = _dbt_project_dir(settings)
        run_dbt(project_dir=project_dir, duckdb_path=settings.duckdb_path, command="build")
        with open_duckdb(settings.duckdb_path) as conn:
            row = conn.execute("SELECT count(*) FROM main_marts.fct_campaign_daily").fetchone()
            assert row is not None
            counter.rows = int(row[0])
        context.add_output_metadata({"dbt_project": MetadataValue.path(str(project_dir))})


@asset(
    deps=[dbt_marts],
    required_resource_keys={"settings"},
    description="Aggregated marts upserted into the Postgres serving layer",
)
def campaign_daily_mart(context: AssetExecutionContext) -> None:
    """Upsert the DuckDB fact table and the campaign dimension into Postgres.

    The dimension loads here rather than in an asset of its own for two
    reasons. It is the same responsibility — both calls take this run's
    DuckDB marts and full-refresh a Postgres serving table — and a separate
    asset would open the DuckDB file a second time, which under Dagster's
    multiprocess executor means a second concurrent reader of a file that
    permits only one writer.

    `load_dim_campaign` existed, was tested, and was read by the API's
    campaign and anomaly queries, but nothing in the pipeline ever called it,
    so every real run left `dim_campaign` empty and the dashboard rendered
    identifiers where names belong. The dimension tests passed throughout,
    because the fixtures that build them call the loader themselves.
    """
    settings: Settings = context.resources.settings
    with _record(context, settings) as counter:
        loaded = load_mart_to_postgres(settings.duckdb_path, settings.postgres_dsn)
        dimension = load_dim_campaign(settings.duckdb_path, settings.postgres_dsn)
        counter.rows = loaded + dimension
        context.add_output_metadata(
            {
                "rows_upserted": MetadataValue.int(loaded),
                "dimension_rows": MetadataValue.int(dimension),
            }
        )


@asset(
    deps=[campaign_daily_mart],
    required_resource_keys={"settings"},
    description="Trained anomaly detector and triage classifier, logged to MLflow",
)
def trained_models(context: AssetExecutionContext) -> None:
    """Train both models on a temporal split of the mart and record the run.

    Feature building, the split, fitting, and evaluation all live in
    `adsignal.ml.training.train_and_evaluate` - the same recipe `adsignal.cli.quality_gate`
    uses, so the CI gate can never certify a model trained differently from
    the one this asset produces (I7).
    """
    settings: Settings = context.resources.settings

    with _record(context, settings) as counter:
        result = train_and_evaluate(read_campaign_daily(settings.postgres_dsn), seed=settings.seed)

        run_id, artifact_dir = log_training_run(
            tracking_dir=settings.mlflow_dir,
            detector=result.detector,
            classifier=result.classifier,
            triage_metrics=result.triage_metrics,
            detector_metrics=result.detector_metrics,
            params={
                "baseline_window": BASELINE_WINDOW,
                "holdout_fraction": HOLDOUT_FRACTION,
                "seed": settings.seed,
            },
        )
        write_model_metrics(
            settings.postgres_dsn,
            run_id,
            datetime.now(UTC),
            result.triage_metrics,
            result.detector_metrics,
            str(artifact_dir),
        )
        _write_artifact_store(settings.postgres_dsn, run_id, result.detector, result.classifier)

        counter.rows = len(result.train) + len(result.holdout)
        context.add_output_metadata(
            {
                "run_id": MetadataValue.text(run_id),
                "triage_macro_f1": MetadataValue.float(result.triage_metrics.macro_f1),
                "triage_accuracy": MetadataValue.float(result.triage_metrics.accuracy),
                "detector_roc_auc": MetadataValue.float(result.detector_metrics.roc_auc),
                "train_rows": MetadataValue.int(len(result.train)),
                "holdout_rows": MetadataValue.int(len(result.holdout)),
            }
        )


def _write_artifact_store(
    dsn: str, run_id: str, detector: AnomalyDetector, classifier: TriageClassifier
) -> None:
    """Persist this run's fitted models to Postgres, the API's serving source.

    Alongside `log_training_run`'s filesystem write, not instead of it: the
    filesystem copy keeps `dagster dev` and existing tooling working, and
    costs nothing extra to also write. The database copy is what makes
    `POST /score` answerable on a host where the API and this pipeline do not
    share a disk - see `adsignal.warehouse.artifact_store`.
    """
    detector_bytes, triage_bytes = dumps_models(detector, classifier)
    write_artifacts(
        dsn,
        run_id,
        detector_bytes,
        triage_bytes,
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
    )


@asset(
    deps=[trained_models],
    required_resource_keys={"settings"},
    description="Per-campaign-day anomaly scores and triage predictions in Postgres",
)
def anomaly_scores(context: AssetExecutionContext) -> None:
    """Score every campaign-day and upsert the predictions."""
    settings: Settings = context.resources.settings
    with _record(context, settings) as counter:
        run_id, artifact_dir = _latest_run(settings)

        features = build_features(
            read_campaign_daily(settings.postgres_dsn), baseline_window=BASELINE_WINDOW
        )
        detector, classifier = load_models(artifact_dir)

        flagged = features[detector.predict(features)]
        if flagged.empty:
            context.add_output_metadata({"rows_written": MetadataValue.int(0)})
            return

        predictions = classifier.predict(flagged)
        predictions["event_date"] = flagged["event_date"].to_numpy()
        predictions["campaign_id"] = flagged["campaign_id"].to_numpy()
        predictions["anomaly_score"] = detector.score(flagged)

        written = write_predictions(settings.postgres_dsn, run_id, predictions)
        counter.rows = written
        context.add_output_metadata({"rows_written": MetadataValue.int(written)})


@asset(
    deps=[trained_models],
    required_resource_keys={"settings"},
    description="Feature drift between the training window and the most recent window",
)
def drift_report(context: AssetExecutionContext) -> None:
    """Compare the holdout window's feature distributions against the training window."""
    settings: Settings = context.resources.settings
    with _record(context, settings) as counter:
        run_id, _ = _latest_run(settings)

        features = build_features(
            read_campaign_daily(settings.postgres_dsn), baseline_window=BASELINE_WINDOW
        )
        reference, current = temporal_split(features, holdout_fraction=HOLDOUT_FRACTION)

        reports = feature_drift(reference, current, list(DETECTOR_FEATURES))
        write_drift_report(settings.postgres_dsn, run_id, datetime.now(UTC), reports)

        counter.rows = len(reports)
        context.add_output_metadata(
            {
                "features_measured": MetadataValue.int(len(reports)),
                "features_drifted": MetadataValue.int(sum(r.drifted for r in reports)),
            }
        )


def _latest_run(settings: Settings) -> tuple[str, Path]:
    """Return the most recent training run's id and artifact directory.

    The run registry (this Postgres table, shared across processes) and the
    artifact store (a directory under this process's `settings.mlflow_dir`)
    are two independent sources of truth. Reading `artifact_uri` from the
    same row as `run_id` - rather than reconstructing
    `settings.mlflow_dir / "artifacts" / run_id` and hoping it matches -
    is what keeps them from silently disagreeing (I4): a run trained under a
    different `mlflow_dir` (a different container, a test's `tmp_path`, a
    concurrent run) is caught here with a clear error instead of surfacing as
    a `FileNotFoundError` three calls downstream in `load_models`.
    """
    with psycopg.connect(settings.postgres_dsn) as conn:
        row = conn.execute(
            "SELECT run_id, artifact_uri FROM model_metrics "
            "ORDER BY trained_at DESC, id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("no training run recorded; materialize trained_models first")
    run_id, artifact_uri = str(row[0]), Path(str(row[1]))
    if not artifact_uri.exists():
        raise RuntimeError(
            f"training run {run_id!r} recorded artifact_uri {artifact_uri}, which does not "
            f"exist under this process's settings (mlflow_dir={settings.mlflow_dir}); the run "
            "was likely trained with a different ADSIGNAL_MLFLOW_DIR, or its artifacts were "
            "cleaned up separately from its model_metrics row"
        )
    return run_id, artifact_uri


@asset(
    deps=[dbt_marts],
    required_resource_keys={"settings"},
    description="Embeddings for campaign names and creative headlines, in pgvector",
)
def campaign_embeddings(context: AssetExecutionContext) -> None:
    """Embed staging text and upsert the vectors into Postgres."""
    settings: Settings = context.resources.settings

    with _record(context, settings) as counter:
        entities = read_entity_text(settings.duckdb_path)
        if not entities:
            context.add_output_metadata({"rows_written": MetadataValue.int(0)})
            return

        model = EmbeddingModel()
        vectors = model.embed([entity.text for entity in entities])

        written = upsert_embeddings(
            settings.postgres_dsn,
            [
                EmbeddingRow(entity.entity_kind, entity.entity_id, entity.text, vector)
                for entity, vector in zip(entities, vectors, strict=True)
            ],
        )
        counter.rows = written
        context.add_output_metadata(
            {
                "rows_written": MetadataValue.int(written),
                "campaigns": MetadataValue.int(
                    sum(1 for e in entities if e.entity_kind == "campaign")
                ),
                "creatives": MetadataValue.int(
                    sum(1 for e in entities if e.entity_kind == "creative")
                ),
            }
        )


@asset(
    deps=[anomaly_scores],
    required_resource_keys={"settings", "narrative_chain"},
    description="A generated prose narrative summarizing the most recent day's anomalies",
)
def daily_narrative(context: AssetExecutionContext) -> None:
    """Generate and store a narrative for the most recently scored day.

    A missing narrative is a degraded dashboard, not a failed pipeline:
    `AllProvidersFailed` - raised when every provider has failed, or when the
    chain's total budget ran out before one answered - is caught here,
    logged, and this asset returns without raising, so a narrative outage
    never blocks the nightly job or any other downstream asset.

    The chain arrives as a resource rather than being built here. A chain
    built per materialisation carries a fresh `CircuitBreaker` per provider,
    and this asset calls `generate` exactly once, so `is_open()` was always
    False and the breakers had no effect on the deployed system at all: a
    permanently dead provider was retried in full, with its timeout, on every
    nightly run. One chain per process is what makes that state real.
    """
    settings: Settings = context.resources.settings
    chain: NarrativeChain = context.resources.narrative_chain

    with _record(context, settings) as counter:
        request = _build_narrative_request(settings.postgres_dsn)
        if request is None:
            context.log.warning("no campaign-daily rows found; skipping narrative generation")
            return

        try:
            result = chain.generate(request)
        except AllProvidersFailed as error:
            # Only the provider names and exception *type* are logged, never the
            # exception object itself: `NarrativeChain` composes its failure
            # message the same way precisely so a provider's error detail (which
            # could echo back an API key) never reaches a log line or a stored
            # row, and logging `error` directly here would undo that.
            context.log.warning(f"narrative generation failed: {type(error).__name__}")
            return

        write_narrative(settings.postgres_dsn, request.as_of, result)
        counter.rows = 1
        context.add_output_metadata(
            {
                "provider": MetadataValue.text(result.provider),
                "latency_ms": MetadataValue.float(result.latency_ms),
            }
        )


def _build_narrative_request(dsn: str) -> NarrativeRequest | None:
    """Summarize the most recently scored day into a `NarrativeRequest`.

    `as_of` is anchored on the campaign-daily mart rather than on
    `anomaly_predictions`: the mart always has rows once the pipeline has
    run, while a quiet day can legitimately leave `anomaly_predictions` empty
    for that date - anchoring there would silently lose the narrative on the
    one day it would say "all quiet".
    """
    with psycopg.connect(dsn) as conn:
        max_date_row = conn.execute("SELECT max(event_date) FROM fct_campaign_daily").fetchone()
        as_of = max_date_row[0] if max_date_row is not None else None
        if as_of is None:
            return None

        campaign_count_row = conn.execute(
            "SELECT count(DISTINCT campaign_id) FROM fct_campaign_daily WHERE event_date = %s",
            (as_of,),
        ).fetchone()
        assert campaign_count_row is not None
        campaign_count = campaign_count_row[0]

        anomaly_row = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE escalate) FROM anomaly_predictions "
            "WHERE event_date = %s",
            (as_of,),
        ).fetchone()
        assert anomaly_row is not None
        anomaly_count, escalated_count = anomaly_row

        cause_rows = conn.execute(
            "SELECT cause FROM anomaly_predictions WHERE event_date = %s "
            "GROUP BY cause ORDER BY count(*) DESC, cause",
            (as_of,),
        ).fetchall()

    return NarrativeRequest(
        as_of=as_of,
        campaign_count=int(campaign_count),
        anomaly_count=int(anomaly_count),
        top_causes=tuple(row[0] for row in cause_rows),
        escalated_count=int(escalated_count),
    )
