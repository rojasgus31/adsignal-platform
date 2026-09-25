"""Deriving the API's two campaign projections from the mart.

Both projections are answered by a query shaped to the response: the campaign
list is a `DISTINCT ON` over the mart's key, and a campaign's series is a
predicate on `campaign_id`. Neither reads the mart into pandas — request cost
and peak memory then scale with the mart's total size rather than with the
~96 rows or one series actually returned, which is an out-of-memory path
inside the API container's cap.

Parameters are always bound by psycopg, never interpolated, like every other
query in `warehouse/`.
"""

from dataclasses import dataclass
from datetime import date

import psycopg

from adsignal.warehouse.loader import ensure_schema

# The mart is keyed (event_date, campaign_id), so a campaign's advertiser and
# channel are carried on every one of its rows. `DISTINCT ON` takes them from
# the campaign's earliest row and lets Postgres do the deduplication.
_LIST_CAMPAIGNS_SQL = """
SELECT DISTINCT ON (campaign_id) campaign_id, advertiser_id, channel
FROM fct_campaign_daily
ORDER BY campaign_id, event_date
"""

_CAMPAIGN_METRICS_SQL = """
SELECT event_date, impressions, clicks, conversions, spend, revenue, is_anomaly
FROM fct_campaign_daily
WHERE campaign_id = %s
ORDER BY event_date
"""


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

    Empty, not an error, on a mart that has never been loaded — including
    against a database where `fct_campaign_daily` does not exist at all yet,
    which `ensure_schema` guards against before the query runs.
    """
    ensure_schema(dsn)
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(_LIST_CAMPAIGNS_SQL).fetchall()
    return [
        CampaignInfo(str(campaign_id), str(advertiser_id), str(channel))
        for campaign_id, advertiser_id, channel in rows
    ]


def read_campaign_metrics(dsn: str, campaign_id: str) -> list[CampaignMetricPoint]:
    """Return one campaign's daily rows, ordered by date.

    Empty (not an error) if the campaign is unknown, or if
    `fct_campaign_daily` does not exist yet; the router turns an empty result
    into a 404.
    """
    ensure_schema(dsn)
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(_CAMPAIGN_METRICS_SQL, (campaign_id,)).fetchall()
    return [
        CampaignMetricPoint(
            event_date=event_date,
            impressions=int(impressions),
            clicks=int(clicks),
            conversions=int(conversions),
            spend=float(spend),
            revenue=float(revenue),
            is_anomaly=bool(is_anomaly),
        )
        for event_date, impressions, clicks, conversions, spend, revenue, is_anomaly in rows
    ]
