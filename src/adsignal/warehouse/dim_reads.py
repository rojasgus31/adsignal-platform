"""Reads of the campaign dimension.

Kept separate from `campaign_reads.py` because a dimension is small, static
within a pipeline run, and read on nearly every request — a different access
pattern from the fact table's filtered time series.
"""

from dataclasses import dataclass

from adsignal.warehouse.loader import ensure_schema
from adsignal.warehouse.pool import connection

_SELECT = """
SELECT campaign_id, campaign_name, advertiser_id, advertiser_name, channel
FROM dim_campaign
ORDER BY campaign_id
"""


@dataclass(frozen=True, slots=True)
class CampaignDimension:
    """One campaign's display identity."""

    campaign_id: str
    campaign_name: str
    advertiser_id: str
    advertiser_name: str
    channel: str


def read_campaign_dimension(dsn: str) -> list[CampaignDimension]:
    """Every campaign's names, or an empty list on a database with no rows."""
    ensure_schema(dsn)
    with connection(dsn) as conn:
        rows = conn.execute(_SELECT).fetchall()
    return [CampaignDimension(*row) for row in rows]
