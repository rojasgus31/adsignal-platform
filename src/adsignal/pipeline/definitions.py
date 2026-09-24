"""Dagster entry point. `dagster dev -m adsignal.pipeline.definitions` runs the UI."""

from dagster import Definitions, ScheduleDefinition, define_asset_job

from adsignal.config import get_settings
from adsignal.pipeline.assets import (
    anomaly_scores,
    campaign_daily_mart,
    catalog_and_raw_events,
    dbt_marts,
    drift_report,
    trained_models,
)

adsignal_daily = define_asset_job(
    name="adsignal_daily",
    selection=[
        catalog_and_raw_events,
        dbt_marts,
        campaign_daily_mart,
        trained_models,
        anomaly_scores,
        drift_report,
    ],
)

daily_schedule = ScheduleDefinition(
    job=adsignal_daily,
    cron_schedule="0 4 * * *",
    execution_timezone="UTC",
)

defs = Definitions(
    assets=[
        catalog_and_raw_events,
        dbt_marts,
        campaign_daily_mart,
        trained_models,
        anomaly_scores,
        drift_report,
    ],
    jobs=[adsignal_daily],
    schedules=[daily_schedule],
    resources={"settings": get_settings()},
)
