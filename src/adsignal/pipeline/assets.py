"""Dagster assets for the AdSignal data platform.

Each asset is idempotent: re-materializing it produces the same end state.
"""

from pathlib import Path

from dagster import AssetExecutionContext, MetadataValue, asset

from adsignal.config import Settings
from adsignal.generator.anomalies import AnomalyInjector
from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events
from adsignal.warehouse.duckdb_io import open_duckdb, register_sources, run_dbt
from adsignal.warehouse.loader import load_mart_to_postgres
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
