import os

import psycopg
import pytest

from adsignal.ml.embeddings import EmbeddingModel
from adsignal.warehouse.semantic_search import semantic_search
from adsignal.warehouse.vector_store import EmbeddingRow, ensure_vector_schema, upsert_embeddings

DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    return EmbeddingModel()


@pytest.fixture
def seeded(model: EmbeddingModel):
    ensure_vector_schema(DSN)
    with psycopg.connect(DSN) as conn:
        conn.execute("TRUNCATE TABLE entity_embeddings")
        conn.commit()

    texts = {
        "cmp_winter": "winter coat clearance discount",
        "cmp_audit": "quarterly financial audit compliance",
        "cmp_summer": "summer sandals promotion sale",
    }
    vectors = model.embed(list(texts.values()))
    upsert_embeddings(
        DSN,
        [
            EmbeddingRow("campaign", entity_id, text, vector)
            for (entity_id, text), vector in zip(texts.items(), vectors, strict=True)
        ],
    )
    return texts


def test_search_finds_the_semantically_closest_campaign(seeded, model) -> None:
    hits = semantic_search(DSN, "jacket sale for cold weather", model, limit=1)
    assert hits[0].entity_id == "cmp_winter"


def test_search_does_not_require_a_literal_word_match(seeded, model) -> None:
    """No word of the query appears in the stored text — this is the point."""
    hits = semantic_search(DSN, "beachwear bargain", model, limit=1)
    assert hits[0].entity_id == "cmp_summer"


def test_limit_is_honored(seeded, model) -> None:
    assert len(semantic_search(DSN, "sale", model, limit=2)) == 2


def test_results_are_ordered_nearest_first(seeded, model) -> None:
    hits = semantic_search(DSN, "cold weather clothing deal", model, limit=3)
    distances = [hit.distance for hit in hits]
    assert distances == sorted(distances)


def test_blank_query_raises(seeded, model) -> None:
    with pytest.raises(ValueError, match="empty"):
        semantic_search(DSN, "   ", model)


def test_search_against_an_empty_store_returns_empty(model) -> None:
    ensure_vector_schema(DSN)
    with psycopg.connect(DSN) as conn:
        conn.execute("TRUNCATE TABLE entity_embeddings")
        conn.commit()
    assert semantic_search(DSN, "anything at all", model) == []
