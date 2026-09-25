from adsignal.warehouse.duckdb_io import open_duckdb
from adsignal.warehouse.text_source import EntityText, read_entity_text


def test_reads_both_campaign_and_creative_text(built_duckdb) -> None:
    rows = read_entity_text(built_duckdb)
    kinds = {row.entity_kind for row in rows}
    assert kinds == {"campaign", "creative"}


def test_every_row_carries_non_empty_text(built_duckdb) -> None:
    rows = read_entity_text(built_duckdb)
    assert rows
    assert all(row.text.strip() for row in rows)


def test_entity_ids_are_unique_within_a_kind(built_duckdb) -> None:
    rows = read_entity_text(built_duckdb)
    keys = [(row.entity_kind, row.entity_id) for row in rows]
    assert len(keys) == len(set(keys))


def test_campaign_text_includes_the_campaign_name(built_duckdb) -> None:
    """`len(text) > len(id)` alone would pass for any suffix — even a hardcoded
    placeholder appended to the id. Check the actual campaign name, queried
    from the same DuckDB file `read_entity_text` reads from, appears in it."""
    with open_duckdb(built_duckdb) as conn:
        expected_names = dict(
            conn.execute(
                "SELECT campaign_id, campaign_name FROM main_staging.stg_campaigns"
            ).fetchall()
        )

    campaigns = {
        row.entity_id: row
        for row in read_entity_text(built_duckdb)
        if row.entity_kind == "campaign"
    }
    assert expected_names
    for campaign_id, name in expected_names.items():
        assert name in campaigns[campaign_id].text


def test_returns_entity_text_instances(built_duckdb) -> None:
    rows = read_entity_text(built_duckdb)
    assert all(isinstance(row, EntityText) for row in rows)
