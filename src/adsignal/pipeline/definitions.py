"""Dagster entry point. `dagster dev -m adsignal.pipeline.definitions` runs the UI."""

from dagster import Definitions, ScheduleDefinition, define_asset_job

from adsignal.config import get_settings
from adsignal.narrative.build import build_chain
from adsignal.pipeline.assets import (
    anomaly_scores,
    campaign_daily_mart,
    campaign_embeddings,
    catalog_and_raw_events,
    daily_narrative,
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
        campaign_embeddings,
        daily_narrative,
    ],
)

daily_schedule = ScheduleDefinition(
    job=adsignal_daily,
    cron_schedule="0 4 * * *",
    execution_timezone="UTC",
)

# One `Settings` and one `NarrativeChain` for the whole process. The chain is
# deliberately longer-lived than a materialisation: its per-provider circuit
# breakers are state, and state rebuilt per run is state that never fires.
_settings = get_settings()

defs = Definitions(
    assets=[
        catalog_and_raw_events,
        dbt_marts,
        campaign_daily_mart,
        trained_models,
        anomaly_scores,
        drift_report,
        campaign_embeddings,
        daily_narrative,
    ],
    jobs=[adsignal_daily],
    schedules=[daily_schedule],
    resources={"settings": _settings, "narrative_chain": build_chain(_settings)},
)
