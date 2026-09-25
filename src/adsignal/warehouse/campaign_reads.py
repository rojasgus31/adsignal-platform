"""Deriving the API's two campaign projections from the mart.

`reader.py` reads the whole mart, unfiltered by campaign. This module builds
the campaign list and the one-campaign time series the API needs on top of
it, so that shaping logic lives in `warehouse/`, not in a router, and so the
API deals in plain typed rows rather than a pandas DataFrame.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import pandas as pd

from adsignal.warehouse.reader import read_campaign_daily


@dataclass(frozen=True, slots=True)
class CampaignInfo:
    """One distinct campaign known to the mart."""

    campaign_id: str
    advertiser_id: str
    channel: str


@dataclass(frozen=True, slots=True)
class CampaignMetricPoint:
    """One day of a campaign's metric time series."""

    event_date: date
    impressions: int
    clicks: int
    conversions: int
    spend: float
    revenue: float
    is_anomaly: bool


def list_campaigns(dsn: str) -> list[CampaignInfo]:
    """Return one entry per distinct campaign, ordered by campaign_id.

    Empty, not an error, on a mart that has never been loaded.
    """
    frame = read_campaign_daily(dsn)
    if frame.empty:
        return []
    unique = (
        frame[["campaign_id", "advertiser_id", "channel"]]
        .drop_duplicates(subset="campaign_id")
        .sort_values("campaign_id")
    )
    return [
        CampaignInfo(str(row.campaign_id), str(row.advertiser_id), str(row.channel))
        for row in unique.itertuples()
    ]


def read_campaign_metrics(dsn: str, campaign_id: str) -> list[CampaignMetricPoint]:
    """Return one campaign's daily rows, ordered by date.

    Empty (not an error) if the campaign is unknown; the router turns that
    into a 404.
    """
    frame = read_campaign_daily(dsn)
    matched = frame[frame["campaign_id"] == campaign_id].sort_values("event_date")
    return [
        CampaignMetricPoint(
            event_date=pd.Timestamp(cast(Any, row.event_date)).date(),
            impressions=int(cast(Any, row.impressions)),
            clicks=int(cast(Any, row.clicks)),
            conversions=int(cast(Any, row.conversions)),
            spend=float(cast(Any, row.spend)),
            revenue=float(cast(Any, row.revenue)),
            is_anomaly=bool(row.is_anomaly),
        )
        for row in matched.itertuples()
    ]
