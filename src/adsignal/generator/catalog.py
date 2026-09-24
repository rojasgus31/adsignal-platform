"""Deterministic construction of the dimension entities."""

import random
from datetime import date, timedelta

from adsignal.generator.models import (
    CHANNELS,
    CREATIVE_FORMATS,
    VERTICALS,
    Advertiser,
    Campaign,
    Catalog,
    Creative,
)

_BRAND_WORDS = (
    "Nimbus",
    "Vertex",
    "Harbor",
    "Lumen",
    "Cobalt",
    "Juniper",
    "Meridian",
    "Quarry",
    "Solstice",
    "Thicket",
    "Vantage",
    "Wren",
)
_BRAND_SUFFIXES = ("Group", "Labs", "Co", "Collective", "Partners")
_CAMPAIGN_THEMES = (
    "Always On",
    "Spring Launch",
    "Summer Push",
    "Back to School",
    "Holiday Retail",
    "Brand Lift",
    "Retargeting",
    "Prospecting",
)
_HEADLINES = (
    "Made for the everyday",
    "See the difference",
    "Built to last",
    "Your next favorite",
    "Now in more places",
    "Start today",
)


def build_catalog(
    *,
    seed: int,
    start_date: date,
    end_date: date,
    n_advertisers: int = 12,
    campaigns_per_advertiser: int = 8,
    creatives_per_campaign: int = 4,
) -> Catalog:
    """Build advertisers, campaigns, and creatives deterministically from a seed.

    Every campaign's date range is contained within [start_date, end_date].
    """
    rng = random.Random(seed)
    window_days = (end_date - start_date).days

    advertisers: list[Advertiser] = []
    campaigns: list[Campaign] = []
    creatives: list[Creative] = []

    for advertiser_index in range(n_advertisers):
        advertiser_id = f"adv_{advertiser_index:04d}"
        name = f"{rng.choice(_BRAND_WORDS)} {rng.choice(_BRAND_SUFFIXES)}"
        advertisers.append(Advertiser(id=advertiser_id, name=name, vertical=rng.choice(VERTICALS)))

        for campaign_index in range(campaigns_per_advertiser):
            campaign_id = f"cmp_{advertiser_index:04d}_{campaign_index:03d}"
            offset = rng.randint(0, max(window_days - 30, 0))
            duration = rng.randint(30, 180)
            campaign_start = start_date + timedelta(days=offset)
            campaign_end = min(campaign_start + timedelta(days=duration), end_date)
            campaigns.append(
                Campaign(
                    id=campaign_id,
                    advertiser_id=advertiser_id,
                    name=f"{name} — {rng.choice(_CAMPAIGN_THEMES)}",
                    channel=rng.choice(CHANNELS),
                    start_date=campaign_start,
                    end_date=campaign_end,
                    daily_budget=round(rng.uniform(200.0, 5000.0), 2),
                )
            )

            for creative_index in range(creatives_per_campaign):
                creatives.append(
                    Creative(
                        id=f"crv_{advertiser_index:04d}_{campaign_index:03d}_{creative_index:02d}",
                        campaign_id=campaign_id,
                        headline=rng.choice(_HEADLINES),
                        format=rng.choice(CREATIVE_FORMATS),
                    )
                )

    return Catalog(
        advertisers=tuple(advertisers),
        campaigns=tuple(campaigns),
        creatives=tuple(creatives),
    )
