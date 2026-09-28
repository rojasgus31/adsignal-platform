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

from adsignal.warehouse.loader import ensure_schema
from adsignal.warehouse.pool import connection

# The mart is keyed (event_date, campaign_id), so a campaign's advertiser and
# channel are carried on every one of its rows. `DISTINCT ON` takes them from
# the campaign's earliest row and lets Postgres do the deduplication.
#
# The display names come from `dim_campaign` in the same query. It is a LEFT
# JOIN, and the names fall back to the identifiers, because nothing in the
# schema forces a fact campaign to have a dimension row: a pipeline that
# loaded the mart but not the dimension would otherwise drop those campaigns
# from the list entirely. One join, not one lookup per row.
#
# `NULLIF(..., '')` sits inside each `COALESCE` because `COALESCE` substitutes
# only on SQL NULL, while `dim_campaign.campaign_name` is `TEXT NOT NULL` and
# its dbt `not_null` test both admit the empty string. Without it a dimension
# row carrying '' would serve '' - an empty cell in the dashboard, which is
# the outcome this fallback exists to prevent. A blank name is treated as a
# missing one.
_LIST_CAMPAIGNS_SQL = """
SELECT facts.campaign_id,
       COALESCE(NULLIF(dim.campaign_name, ''), facts.campaign_id) AS campaign_name,
       facts.advertiser_id,
       COALESCE(NULLIF(dim.advertiser_name, ''), facts.advertiser_id) AS advertiser_name,
       facts.channel
FROM (
    SELECT DISTINCT ON (campaign_id) campaign_id, advertiser_id, channel
    FROM fct_campaign_daily
    ORDER BY campaign_id, event_date
) AS facts
LEFT JOIN dim_campaign AS dim ON dim.campaign_id = facts.campaign_id
ORDER BY facts.campaign_id
"""

_CAMPAIGN_METRICS_SQL = """
SELECT event_date, impressions, clicks, conversions, spend, revenue, is_anomaly
FROM fct_campaign_daily
WHERE campaign_id = %s
ORDER BY event_date
"""


@dataclass(frozen=True, slots=True)
class CampaignInfo:
    """One distinct campaign known to the mart, with its display names.

    `campaign_name` and `advertiser_name` hold the campaign's own identifier
    and its advertiser's identifier when `dim_campaign` has no row for the
    campaign, or when the row it has carries a blank name - see
    `_LIST_CAMPAIGNS_SQL`. They are never None.

    They are not guaranteed non-empty: that would require the identifiers
    themselves to be non-empty, and `campaign_id`/`advertiser_id` are only
    `TEXT NOT NULL` in the mart. What is guaranteed is that a blank in the
    dimension is never what gets served.
    """

    campaign_id: str
    campaign_name: str
    advertiser_id: str
    advertiser_name: str
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
    against a database where neither `fct_campaign_daily` nor `dim_campaign`
    exists yet: `ensure_schema` creates both before the query runs.
    """
    ensure_schema(dsn)
    with connection(dsn) as conn:
        rows = conn.execute(_LIST_CAMPAIGNS_SQL).fetchall()
    return [
        CampaignInfo(
            str(campaign_id),
            str(campaign_name),
            str(advertiser_id),
            str(advertiser_name),
            str(channel),
        )
        for campaign_id, campaign_name, advertiser_id, advertiser_name, channel in rows
    ]


def read_campaign_metrics(dsn: str, campaign_id: str) -> list[CampaignMetricPoint]:
    """Return one campaign's daily rows, ordered by date.

    Empty (not an error) if the campaign is unknown, or if
    `fct_campaign_daily` does not exist yet; the router turns an empty result
    into a 404.
    """
    ensure_schema(dsn)
    with connection(dsn) as conn:
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
