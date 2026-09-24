import os

import psycopg
import pytest

from adsignal.warehouse.vector_store import (
    EMBEDDING_DIM,
    EmbeddingRow,
    ensure_vector_schema,
    search_similar,
    upsert_embeddings,
)

DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


def _vector(seed: float) -> list[float]:
    """A deterministic unit-ish vector of the declared width."""
    return [seed] + [0.0] * (EMBEDDING_DIM - 1)


@pytest.fixture(autouse=True)
def clean_vectors():
    ensure_vector_schema(DSN)
    with psycopg.connect(DSN) as conn:
        conn.execute("TRUNCATE TABLE entity_embeddings")
        conn.commit()
    yield


def test_ensure_schema_is_idempotent() -> None:
    ensure_vector_schema(DSN)
    ensure_vector_schema(DSN)


def test_upsert_then_search_returns_the_nearest_row() -> None:
    upsert_embeddings(
        DSN,
        [
            EmbeddingRow("campaign", "cmp_a", "spring sale", _vector(1.0)),
            EmbeddingRow("campaign", "cmp_b", "winter clearance", _vector(-1.0)),
        ],
    )
    hits = search_similar(DSN, _vector(1.0), limit=1)
    assert len(hits) == 1
    assert hits[0].entity_id == "cmp_a"
    assert hits[0].text == "spring sale"


def test_search_orders_by_increasing_distance() -> None:
    upsert_embeddings(
        DSN,
        [
            EmbeddingRow("campaign", "near", "a", _vector(1.0)),
            EmbeddingRow("campaign", "far", "b", _vector(-1.0)),
        ],
    )
    hits = search_similar(DSN, _vector(1.0), limit=2)
    assert [hit.entity_id for hit in hits] == ["near", "far"]
    assert hits[0].distance <= hits[1].distance


def test_upsert_replaces_rather_than_duplicating() -> None:
    row = EmbeddingRow("campaign", "cmp_a", "first", _vector(1.0))
    assert upsert_embeddings(DSN, [row]) == 1
    assert upsert_embeddings(DSN, [EmbeddingRow("campaign", "cmp_a", "second", _vector(1.0))]) == 1
    with psycopg.connect(DSN) as conn:
        count = conn.execute("SELECT count(*) FROM entity_embeddings").fetchone()[0]
        text = conn.execute("SELECT text FROM entity_embeddings").fetchone()[0]
    assert count == 1
    assert text == "second"


def test_kind_filter_restricts_results() -> None:
    upsert_embeddings(
        DSN,
        [
            EmbeddingRow("campaign", "cmp_a", "a", _vector(1.0)),
            EmbeddingRow("creative", "crv_a", "b", _vector(1.0)),
        ],
    )
    hits = search_similar(DSN, _vector(1.0), limit=10, kind="creative")
    assert [hit.entity_kind for hit in hits] == ["creative"]


def test_search_on_an_empty_table_returns_an_empty_list() -> None:
    assert search_similar(DSN, _vector(1.0)) == []


def test_upsert_rejects_a_wrong_width_vector() -> None:
    """A dimension mismatch must fail loudly at the boundary, not inside Postgres."""
    bad = EmbeddingRow("campaign", "cmp_a", "a", [0.1, 0.2])
    with pytest.raises(ValueError, match=f"expected {EMBEDDING_DIM}"):
        upsert_embeddings(DSN, [bad])


def test_upsert_of_nothing_writes_nothing() -> None:
    assert upsert_embeddings(DSN, []) == 0
