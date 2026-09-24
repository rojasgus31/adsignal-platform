from datetime import date

from adsignal.generator.catalog import build_catalog
from adsignal.generator.models import CHANNELS

START = date(2024, 1, 1)
END = date(2024, 12, 31)


def test_catalog_shape_matches_requested_counts() -> None:
    catalog = build_catalog(
        seed=1,
        n_advertisers=3,
        campaigns_per_advertiser=2,
        creatives_per_campaign=5,
        start_date=START,
        end_date=END,
    )
    assert len(catalog.advertisers) == 3
    assert len(catalog.campaigns) == 6
    assert len(catalog.creatives) == 30


def test_catalog_is_deterministic_for_a_seed() -> None:
    kwargs = dict(
        n_advertisers=4,
        campaigns_per_advertiser=3,
        creatives_per_campaign=2,
        start_date=START,
        end_date=END,
    )
    first = build_catalog(seed=42, **kwargs)
    second = build_catalog(seed=42, **kwargs)
    assert first == second


def test_different_seeds_produce_different_catalogs() -> None:
    kwargs = dict(
        n_advertisers=4,
        campaigns_per_advertiser=3,
        creatives_per_campaign=2,
        start_date=START,
        end_date=END,
    )
    assert build_catalog(seed=1, **kwargs) != build_catalog(seed=2, **kwargs)


def test_every_campaign_falls_inside_the_requested_window() -> None:
    catalog = build_catalog(
        seed=7,
        n_advertisers=5,
        campaigns_per_advertiser=4,
        creatives_per_campaign=2,
        start_date=START,
        end_date=END,
    )
    for campaign in catalog.campaigns:
        assert START <= campaign.start_date <= campaign.end_date <= END
        assert campaign.channel in CHANNELS


def test_foreign_keys_resolve() -> None:
    catalog = build_catalog(
        seed=3,
        n_advertisers=3,
        campaigns_per_advertiser=2,
        creatives_per_campaign=2,
        start_date=START,
        end_date=END,
    )
    advertiser_ids = {a.id for a in catalog.advertisers}
    campaign_ids = {c.id for c in catalog.campaigns}
    assert all(c.advertiser_id in advertiser_ids for c in catalog.campaigns)
    assert all(cr.campaign_id in campaign_ids for cr in catalog.creatives)


def test_catalog_respects_short_window_boundary() -> None:
    """Campaigns fit within a window shorter than 30 days."""
    short_start = date(2024, 1, 1)
    short_end = date(2024, 1, 10)
    catalog = build_catalog(
        seed=5,
        n_advertisers=2,
        campaigns_per_advertiser=2,
        creatives_per_campaign=2,
        start_date=short_start,
        end_date=short_end,
    )
    for campaign in catalog.campaigns:
        assert short_start <= campaign.start_date <= campaign.end_date <= short_end


def test_catalog_respects_single_day_window() -> None:
    """Campaigns fit within a degenerate single-day window."""
    single_day = date(2024, 6, 15)
    catalog = build_catalog(
        seed=6,
        n_advertisers=2,
        campaigns_per_advertiser=2,
        creatives_per_campaign=2,
        start_date=single_day,
        end_date=single_day,
    )
    for campaign in catalog.campaigns:
        assert single_day <= campaign.start_date <= campaign.end_date <= single_day
