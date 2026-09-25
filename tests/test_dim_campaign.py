"""Review Focus 1: a fact row whose campaign has no dimension row."""

import os
from pathlib import Path

import psycopg

from adsignal.warehouse.dim_reads import CampaignDimension, read_campaign_dimension
from adsignal.warehouse.duckdb_io import open_duckdb

TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


def test_every_campaign_has_a_human_name(loaded_dimension) -> None:
    rows = read_campaign_dimension(TEST_DSN)
    assert rows, "the dimension is empty"
    for row in rows:
        assert isinstance(row, CampaignDimension)
        assert row.campaign_name and not row.campaign_name.startswith("cmp_")
        assert row.advertiser_name and not row.advertiser_name.startswith("adv_")


def test_campaign_ids_are_unique(loaded_dimension) -> None:
    """A duplicate would silently multiply every joined fact row."""
    ids = [row.campaign_id for row in read_campaign_dimension(TEST_DSN)]
    assert len(ids) == len(set(ids))


def test_every_fact_campaign_resolves_to_a_dimension_row(loaded_dimension) -> None:
    """Review Focus 1. An unmatched id means the dashboard drops rows or shows undefined."""
    with psycopg.connect(TEST_DSN) as conn:
        orphans = conn.execute(
            "SELECT count(DISTINCT f.campaign_id) FROM fct_campaign_daily f "
            "LEFT JOIN dim_campaign d ON d.campaign_id = f.campaign_id "
            "WHERE d.campaign_id IS NULL"
        ).fetchone()[0]
    assert orphans == 0, f"{orphans} campaign ids in the fact table have no dimension row"


def test_a_campaign_with_no_matching_advertiser_still_appears(
    built_duckdb_with_orphan_campaign: Path,
) -> None:
    """Review Finding 1. An INNER JOIN here would silently drop this campaign
    from `dim_campaign` while it kept fact rows in `fct_campaign_daily`
    (which joins only `stg_campaigns`) — an orphan the other tests in this
    file cannot see, because `build_catalog` never produces one on its own.
    """
    with open_duckdb(built_duckdb_with_orphan_campaign) as conn:
        row = conn.execute(
            "SELECT campaign_name, advertiser_name FROM main_marts.dim_campaign "
            "WHERE campaign_id = 'cmp_orphan'"
        ).fetchone()
    assert row is not None, "the orphan campaign is missing from dim_campaign entirely"
    campaign_name, advertiser_name = row
    assert campaign_name == "Orphan Campaign"
    assert advertiser_name == "Unknown advertiser"


def test_reading_an_empty_dimension_returns_an_empty_list(empty_client) -> None:
    """The fresh-database contract: no tables yet is not an error."""
    dsn = os.environ.get(
        "ADSIGNAL_EMPTY_TEST_DSN",
        "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_empty",
    )
    assert read_campaign_dimension(dsn) == []
