import math

import pytest

from adsignal.ml.embeddings import EMBEDDING_DIM, EmbeddingModel
from adsignal.ml.embeddings import EMBEDDING_DIM as MODEL_DIM
from adsignal.warehouse.vector_store import EMBEDDING_DIM as COLUMN_DIM


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    """Loading the model downloads weights once; share it across the module."""
    return EmbeddingModel()


def test_embedding_dimension_matches_the_declared_constant(model: EmbeddingModel) -> None:
    """Review Focus 3: the vector column is fixed-width.

    If the model's real width ever diverges from the constant, every insert
    fails — or a same-width model from a different space silently returns
    nonsense. Pin the real width against the constant the schema uses.
    """
    vector = model.embed_one("a spring sale campaign")
    assert len(vector) == EMBEDDING_DIM


def test_embed_returns_one_vector_per_input(model: EmbeddingModel) -> None:
    vectors = model.embed(["first text", "second text", "third text"])
    assert len(vectors) == 3
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


def test_embedding_is_deterministic(model: EmbeddingModel) -> None:
    first = model.embed_one("identical text")
    second = model.embed_one("identical text")
    assert first == pytest.approx(second)


def test_similar_texts_are_closer_than_unrelated_ones(model: EmbeddingModel) -> None:
    """The whole point of the model: semantic proximity, not string proximity."""
    anchor = model.embed_one("discount promotion for winter coats")
    similar = model.embed_one("winter jacket sale offer")
    unrelated = model.embed_one("quarterly financial audit procedures")

    assert _cosine(anchor, similar) > _cosine(anchor, unrelated)


def test_embedding_empty_list_returns_empty_list(model: EmbeddingModel) -> None:
    assert model.embed([]) == []


def test_embedding_rejects_an_empty_string(model: EmbeddingModel) -> None:
    with pytest.raises(ValueError, match="empty"):
        model.embed_one("   ")


def test_model_and_column_dimensions_agree() -> None:
    """The two constants are declared separately on purpose — importing
    the model module from the warehouse layer would drag fastembed into
    every vector read. Nothing else keeps them in step, so this does."""
    assert MODEL_DIM == COLUMN_DIM


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
