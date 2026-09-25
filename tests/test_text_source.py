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
    campaigns = [r for r in read_entity_text(built_duckdb) if r.entity_kind == "campaign"]
    assert campaigns
    assert any(len(c.text) > len(c.entity_id) for c in campaigns)


def test_returns_entity_text_instances(built_duckdb) -> None:
    rows = read_entity_text(built_duckdb)
    assert all(isinstance(row, EntityText) for row in rows)
