from datetime import date

import pytest
from dagster import materialize

from adsignal.config import Settings
from adsignal.pipeline.assets import catalog_and_raw_events, dbt_marts


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        target_events=5_000,
        seed=123,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        data_dir=tmp_path,
        duckdb_path=tmp_path / "adsignal.duckdb",
        postgres_dsn="postgresql://adsignal:adsignal@localhost:5432/adsignal",
    )


def test_raw_events_asset_writes_both_parquet_files(settings) -> None:
    result = materialize(
        [catalog_and_raw_events],
        resources={"settings": settings},
    )
    assert result.success
    assert settings.raw_events_path.exists()
    assert settings.anomaly_labels_path.exists()

    metadata = result.asset_materializations_for_node("catalog_and_raw_events")[0].metadata
    assert metadata["row_count"].value > 0
    assert metadata["label_count"].value >= 0


def test_raw_events_asset_is_rerunnable(settings) -> None:
    first = materialize([catalog_and_raw_events], resources={"settings": settings})
    second = materialize([catalog_and_raw_events], resources={"settings": settings})
    assert first.success and second.success

    first_rows = (
        first.asset_materializations_for_node("catalog_and_raw_events")[0]
        .metadata["row_count"]
        .value
    )
    second_rows = (
        second.asset_materializations_for_node("catalog_and_raw_events")[0]
        .metadata["row_count"]
        .value
    )
    assert first_rows == second_rows


def test_dbt_marts_asset_builds_the_fact_table(settings) -> None:
    result = materialize(
        [catalog_and_raw_events, dbt_marts],
        resources={"settings": settings},
    )
    assert result.success

    from adsignal.warehouse.duckdb_io import open_duckdb

    with open_duckdb(settings.duckdb_path) as conn:
        rows = conn.execute("SELECT count(*) FROM main_marts.fct_campaign_daily").fetchone()[0]
    assert rows > 0


def test_definitions_expose_all_three_assets() -> None:
    from adsignal.pipeline.definitions import defs

    keys = {key.to_user_string() for key in defs.resolve_asset_graph().get_all_asset_keys()}
    assert {"catalog_and_raw_events", "dbt_marts", "campaign_daily_mart"} <= keys


def test_catalog_size_scales_with_target_events(tmp_path) -> None:
    """ADSIGNAL_TARGET_EVENTS must actually size the generated catalog.

    Materializing catalog_and_raw_events with a small target_events should
    produce strictly fewer rows than materializing it with a large one.
    Row counts are approximate (advertiser count is derived from an expected
    rows-per-advertiser estimate), so this asserts strict inequality rather
    than an exact row count.
    """
    small_dir = tmp_path / "small"
    large_dir = tmp_path / "large"
    small_dir.mkdir()
    large_dir.mkdir()

    small_settings = Settings(
        target_events=5_000,
        seed=123,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        data_dir=small_dir,
        duckdb_path=small_dir / "adsignal.duckdb",
        postgres_dsn="postgresql://adsignal:adsignal@localhost:5432/adsignal",
    )
    large_settings = Settings(
        target_events=50_000,
        seed=123,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        data_dir=large_dir,
        duckdb_path=large_dir / "adsignal.duckdb",
        postgres_dsn="postgresql://adsignal:adsignal@localhost:5432/adsignal",
    )

    small_result = materialize([catalog_and_raw_events], resources={"settings": small_settings})
    large_result = materialize([catalog_and_raw_events], resources={"settings": large_settings})
    assert small_result.success and large_result.success

    small_rows = (
        small_result.asset_materializations_for_node("catalog_and_raw_events")[0]
        .metadata["row_count"]
        .value
    )
    large_rows = (
        large_result.asset_materializations_for_node("catalog_and_raw_events")[0]
        .metadata["row_count"]
        .value
    )

    assert large_rows > small_rows
