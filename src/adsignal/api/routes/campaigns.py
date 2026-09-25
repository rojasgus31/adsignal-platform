"""`GET /campaigns`, `GET /campaigns/{campaign_id}/metrics`."""

from fastapi import APIRouter, Depends, HTTPException

from adsignal.api.deps import get_settings
from adsignal.api.schemas import CampaignMetrics, CampaignSummary, MetricPoint
from adsignal.config import Settings
from adsignal.warehouse.campaign_reads import list_campaigns, read_campaign_metrics

router = APIRouter(tags=["campaigns"])


@router.get("/campaigns", response_model=list[CampaignSummary])
def get_campaigns(settings: Settings = Depends(get_settings)) -> list[CampaignSummary]:
    """List every campaign known to the mart. Empty before the first pipeline run."""
    return [
        CampaignSummary(
            campaign_id=info.campaign_id,
            advertiser_id=info.advertiser_id,
            channel=info.channel,
        )
        for info in list_campaigns(settings.postgres_dsn)
    ]


@router.get("/campaigns/{campaign_id}/metrics", response_model=CampaignMetrics)
def get_campaign_metrics(
    campaign_id: str, settings: Settings = Depends(get_settings)
) -> CampaignMetrics:
    """Return one campaign's daily time series, or 404 if the campaign is unknown."""
    rows = read_campaign_metrics(settings.postgres_dsn, campaign_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"unknown campaign: {campaign_id}")

    points = [
        MetricPoint(
            event_date=row.event_date,
            impressions=row.impressions,
            clicks=row.clicks,
            conversions=row.conversions,
            spend=row.spend,
            revenue=row.revenue,
            is_anomaly=row.is_anomaly,
        )
        for row in rows
    ]
    return CampaignMetrics(campaign_id=campaign_id, points=points)
