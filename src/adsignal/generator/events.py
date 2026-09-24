"""Deterministic generation of daily delivery rows with seasonality.

Yields lazily so that ten million rows never exist in memory at once.
"""

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

from adsignal.generator.models import Catalog, EventRow

_WEEKDAY_MULTIPLIERS = (1.05, 1.08, 1.10, 1.06, 0.98, 0.78, 0.74)


@dataclass(frozen=True, slots=True)
class ChannelBaseline:
    """Typical delivery economics for one channel."""

    impressions_per_budget_unit: float
    click_through_rate: float
    conversion_rate: float
    revenue_per_conversion: float


CHANNEL_BASELINES: dict[str, ChannelBaseline] = {
    "search": ChannelBaseline(180.0, 0.042, 0.075, 92.0),
    "social": ChannelBaseline(420.0, 0.016, 0.030, 64.0),
    "display": ChannelBaseline(1100.0, 0.0008, 0.011, 48.0),
    "video": ChannelBaseline(650.0, 0.0065, 0.018, 71.0),
    "native": ChannelBaseline(520.0, 0.0041, 0.022, 58.0),
}


def weekday_multiplier(day: date) -> float:
    """Return the weekday demand multiplier. Monday is index 0."""
    return _WEEKDAY_MULTIPLIERS[day.weekday()]


def seasonal_multiplier(day: date) -> float:
    """Return an annual seasonality multiplier in roughly [0.8, 1.2].

    Peaks in late Q4 to mimic holiday retail demand.
    """
    day_of_year = day.timetuple().tm_yday
    phase = 2.0 * math.pi * (day_of_year - 320) / 365.0
    return 1.0 + 0.2 * math.cos(phase)


def generate_events(catalog: Catalog, seed: int) -> Iterator[EventRow]:
    """Yield one EventRow per creative per active campaign-day.

    This is a generator. Callers stream it; nothing accumulates the rows.
    """
    creatives_by_campaign: dict[str, list[str]] = {}
    for creative in catalog.creatives:
        creatives_by_campaign.setdefault(creative.campaign_id, []).append(creative.id)

    for campaign in catalog.campaigns:
        baseline = CHANNEL_BASELINES[campaign.channel]
        creative_ids = creatives_by_campaign.get(campaign.id, [])
        if not creative_ids:
            continue

        share = 1.0 / len(creative_ids)
        total_days = (campaign.end_date - campaign.start_date).days + 1

        for day_offset in range(total_days):
            event_date = campaign.start_date + timedelta(days=day_offset)
            demand = weekday_multiplier(event_date) * seasonal_multiplier(event_date)

            for creative_id in creative_ids:
                # Seeding per row keeps generation deterministic and order
                # independent, which lets the pipeline be resumed or sharded.
                # random.Random only accepts None/int/float/str/bytes/bytearray,
                # so the seed components are joined into a single string.
                row_seed = f"{seed}:{campaign.id}:{creative_id}:{event_date.toordinal()}"
                rng = random.Random(row_seed)
                noise = rng.uniform(0.85, 1.15)

                spend = campaign.daily_budget * share * demand * noise
                impressions = int(spend * baseline.impressions_per_budget_unit / 100.0)
                clicks = _binomial(rng, impressions, baseline.click_through_rate)
                conversions = _binomial(rng, clicks, baseline.conversion_rate)
                revenue = conversions * baseline.revenue_per_conversion * rng.uniform(0.8, 1.25)

                yield EventRow(
                    event_date=event_date,
                    campaign_id=campaign.id,
                    creative_id=creative_id,
                    channel=campaign.channel,
                    impressions=impressions,
                    clicks=clicks,
                    conversions=conversions,
                    spend=round(spend, 4),
                    revenue=round(revenue, 4),
                )


def _binomial(rng: random.Random, trials: int, probability: float) -> int:
    """Draw a binomial sample, approximating normally for large trial counts.

    Exact sampling is O(trials), which is far too slow at ten million rows.
    """
    if trials <= 0 or probability <= 0.0:
        return 0
    if trials < 40:
        return sum(1 for _ in range(trials) if rng.random() < probability)
    mean = trials * probability
    stdev = math.sqrt(trials * probability * (1.0 - probability))
    return max(0, min(trials, round(rng.gauss(mean, stdev))))
