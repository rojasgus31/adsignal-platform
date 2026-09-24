from collections import Counter
from datetime import date, timedelta

from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events, seasonal_multiplier, weekday_multiplier
from adsignal.generator.models import CHANNELS


def _tiny_catalog(start: date, end: date):
    return build_catalog(
        seed=5,
        n_advertisers=2,
        campaigns_per_advertiser=2,
        creatives_per_campaign=2,
        start_date=start,
        end_date=end,
    )


def test_generate_events_is_a_lazy_iterator() -> None:
    catalog = _tiny_catalog(date(2024, 1, 1), date(2024, 3, 31))
    events = generate_events(catalog, seed=1)
    assert not isinstance(events, list)
    assert next(iter(events)) is not None


def test_events_are_deterministic_for_a_seed() -> None:
    catalog = _tiny_catalog(date(2024, 1, 1), date(2024, 2, 29))
    first = list(generate_events(catalog, seed=11))
    second = list(generate_events(catalog, seed=11))
    assert first == second


def test_funnel_is_monotonic() -> None:
    catalog = _tiny_catalog(date(2024, 1, 1), date(2024, 2, 29))
    for row in generate_events(catalog, seed=2):
        assert row.impressions >= row.clicks >= row.conversions >= 0
        assert row.spend >= 0.0
        assert row.revenue >= 0.0


def test_every_row_belongs_to_its_campaign_window() -> None:
    catalog = _tiny_catalog(date(2024, 1, 1), date(2024, 6, 30))
    windows = {c.id: (c.start_date, c.end_date) for c in catalog.campaigns}
    for row in generate_events(catalog, seed=3):
        start, end = windows[row.campaign_id]
        assert start <= row.event_date <= end
        assert row.channel in CHANNELS


def test_leap_year_range_has_every_date_once() -> None:
    """Review Focus 4: a range across Feb 29 and a year boundary must
    produce every calendar day exactly once per creative, with no gaps."""
    start, end = date(2024, 2, 27), date(2024, 3, 2)
    catalog = build_catalog(
        seed=9,
        n_advertisers=1,
        campaigns_per_advertiser=1,
        creatives_per_campaign=1,
        start_date=start,
        end_date=end,
    )
    campaign = catalog.campaigns[0]
    rows = list(generate_events(catalog, seed=4))

    expected_days = {
        campaign.start_date + timedelta(days=offset)
        for offset in range((campaign.end_date - campaign.start_date).days + 1)
    }
    assert date(2024, 2, 29) in expected_days or campaign.end_date < date(2024, 2, 29)

    counts = Counter((row.event_date, row.creative_id) for row in rows)
    assert set(counts.values()) == {1}, "a campaign-day-creative appeared more than once"
    assert {day for day, _ in counts} == expected_days


def test_year_boundary_is_continuous() -> None:
    """Review Focus 4: Dec 31 -> Jan 1 must not drop or duplicate a day."""
    catalog = build_catalog(
        seed=13,
        n_advertisers=1,
        campaigns_per_advertiser=1,
        creatives_per_campaign=1,
        start_date=date(2024, 12, 28),
        end_date=date(2025, 1, 4),
    )
    days = sorted({row.event_date for row in generate_events(catalog, seed=6)})
    # Note: zip(days, days[1:], strict=True) as given in the brief always
    # raises ValueError, since the two slices differ in length by exactly
    # one element for any list of 2+ items. Plain zip() is the correct
    # pairwise idiom here and preserves the intended gap check.
    gaps = [(b - a).days for a, b in zip(days, days[1:], strict=False)]
    assert set(gaps) <= {1}, f"non-contiguous days: {days}"


def test_weekday_multiplier_dips_on_weekends() -> None:
    wednesday = weekday_multiplier(date(2024, 1, 3))
    saturday = weekday_multiplier(date(2024, 1, 6))
    assert saturday < wednesday


def test_seasonal_multiplier_is_bounded() -> None:
    day = date(2024, 1, 1)
    values = [seasonal_multiplier(day + timedelta(days=offset)) for offset in range(366)]
    assert all(0.5 <= value <= 1.5 for value in values)
