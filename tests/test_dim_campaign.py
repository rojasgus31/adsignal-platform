"""Review Focus 1: a fact row whose campaign has no dimension row.

Every test here runs against `loaded_dimension`'s own database (see
`DIM_TEST_DSN` in `conftest.py`), which the fixture loads with both
`fct_campaign_daily` and `dim_campaign` from one DuckDB build. The DSN comes
from the fixture rather than from a module-level constant so that no test in
this file can silently start asserting against another module's rows.

These tests read `dim_campaign` with SQL of their own rather than through a
warehouse helper. `warehouse/dim_reads.py` used to provide one, but Task 9b
made the serving layer read the dimension through a join in
`campaign_reads.list_campaigns` / `serving_reads.read_anomaly_predictions`
instead, which left that module with no caller and no prospect of one; it was
deleted rather than kept alive by its own tests (Task 9b review, Finding 2).
The live read path is covered where it is used - `tests/test_api.py`.
"""

from pathlib import Path

import psycopg

from adsignal.warehouse.campaign_reads import list_campaigns
from adsignal.warehouse.duckdb_io import open_duckdb

_SELECT_DIMENSION = """
SELECT campaign_id, campaign_name, advertiser_id, advertiser_name, channel
FROM dim_campaign
ORDER BY campaign_id
"""


def _dimension_rows(dsn: str) -> list[tuple[str, str, str, str, str]]:
    with psycopg.connect(dsn) as conn:
        return conn.execute(_SELECT_DIMENSION).fetchall()


def test_every_campaign_has_a_human_name(loaded_dimension) -> None:
    rows = _dimension_rows(loaded_dimension)
    assert rows, "the dimension is empty"
    for _, campaign_name, _, advertiser_name, _ in rows:
        assert campaign_name and not campaign_name.startswith("cmp_")
        assert advertiser_name and not advertiser_name.startswith("adv_")


def test_campaign_ids_are_unique(loaded_dimension) -> None:
    """A duplicate would silently multiply every joined fact row."""
    ids = [row[0] for row in _dimension_rows(loaded_dimension)]
    assert len(ids) == len(set(ids))


def test_every_fact_campaign_resolves_to_a_dimension_row(loaded_dimension) -> None:
    """Review Focus 1. An unmatched id means the dashboard drops rows or shows undefined."""
    with psycopg.connect(loaded_dimension) as conn:
        orphans = conn.execute(
            "SELECT count(DISTINCT f.campaign_id) FROM fct_campaign_daily f "
            "LEFT JOIN dim_campaign d ON d.campaign_id = f.campaign_id "
            "WHERE d.campaign_id IS NULL"
        ).fetchone()[0]
    assert orphans == 0, f"{orphans} campaign ids in the fact table have no dimension row"


def test_the_join_returns_one_row_per_distinct_fact_campaign(loaded_dimension) -> None:
    """Task 9b review, Finding 5: the row-count invariant, which every other
    assertion in the suite is blind to because they all look at field values.

    `dim_campaign.campaign_id` is a primary key, so the LEFT JOIN in
    `_LIST_CAMPAIGNS_SQL` can match at most one dimension row per campaign and
    the count is preserved. A duplicate slipping into the dimension would
    inflate this list - and, on `/anomalies`, would shift rows across page
    boundaries - while leaving every name assertion passing.
    """
    with psycopg.connect(loaded_dimension) as conn:
        distinct_fact_campaigns = conn.execute(
            "SELECT count(DISTINCT campaign_id) FROM fct_campaign_daily"
        ).fetchone()[0]

    campaigns = list_campaigns(loaded_dimension)
    assert distinct_fact_campaigns > 0, "the fact table is empty"
    assert len(campaigns) == distinct_fact_campaigns
    assert len({campaign.campaign_id for campaign in campaigns}) == len(campaigns)


def test_a_blank_dimension_name_falls_back_to_the_identifier(loaded_dimension) -> None:
    """Task 9b review, Finding 3. `TEXT NOT NULL` and dbt's `not_null` both
    admit the empty string, so `COALESCE` alone would serve `""` for a
    dimension row with a blank name - an empty cell in the dashboard, which is
    the outcome the fallback exists to prevent. `NULLIF(..., '')` inside the
    `COALESCE` is what makes the blank behave like a missing row.
    """
    campaign_id = list_campaigns(loaded_dimension)[0].campaign_id
    with psycopg.connect(loaded_dimension) as conn:
        conn.execute(
            "UPDATE dim_campaign SET campaign_name = '', advertiser_name = '' "
            "WHERE campaign_id = %s",
            (campaign_id,),
        )
        conn.commit()

    blanked = next(
        campaign
        for campaign in list_campaigns(loaded_dimension)
        if campaign.campaign_id == campaign_id
    )
    assert blanked.campaign_name == campaign_id
    assert blanked.advertiser_name == blanked.advertiser_id


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


def test_the_live_read_path_is_empty_on_a_fresh_database(empty_client) -> None:
    """The fresh-database contract, asserted on the function the API actually
    calls. `read_campaign_dimension` used to stand in for this; it no longer
    exists, and `list_campaigns` is the read whose failure a visitor would
    see.
    """
    dsn = empty_client.app.state.settings.postgres_dsn
    assert list_campaigns(dsn) == []
