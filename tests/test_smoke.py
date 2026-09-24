"""End-to-end check that the three stages compose into one working pipeline."""

from datetime import date

import pytest
from dagster import materialize

from adsignal.config import Settings
from adsignal.pipeline.assets import catalog_and_raw_events, dbt_marts


@pytest.fixture
def tiny_settings(tmp_path) -> Settings:
    return Settings(
        target_events=5_000,
        seed=2024,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        data_dir=tmp_path,
        duckdb_path=tmp_path / "smoke.duckdb",
    )


def test_generate_then_transform_produces_queryable_marts(tiny_settings) -> None:
    result = materialize(
        [catalog_and_raw_events, dbt_marts],
        resources={"settings": tiny_settings},
    )
    assert result.success

    from adsignal.warehouse.duckdb_io import open_duckdb

    with open_duckdb(tiny_settings.duckdb_path) as conn:
        row = conn.execute(
            """
            SELECT count(*) AS days,
                   count(DISTINCT campaign_id) AS campaigns,
                   sum(CASE WHEN is_anomaly THEN 1 ELSE 0 END) AS anomalies
            FROM main_marts.fct_campaign_daily
            """
        ).fetchone()

    days, campaigns, anomalies = row
    assert days > 0
    assert campaigns > 0
    assert anomalies > 0
