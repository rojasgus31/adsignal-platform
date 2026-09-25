import math
import threading
import time

import pytest

from adsignal.ml import embeddings as embeddings_module
from adsignal.ml.embeddings import EMBEDDING_DIM, MODEL_NAME, EmbeddingModel
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


def test_the_model_is_constructed_once_under_concurrent_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding 7: `/search` is a sync def, so Starlette runs it in the anyio
    worker threadpool and several first requests are genuinely concurrent.

    Without synchronisation each of them constructs its own `TextEmbedding`,
    transiently holding several ~90 MB models inside a 1 GB container.
    """
    constructions = []
    construction_lock = threading.Lock()
    workers = 8
    barrier = threading.Barrier(workers)

    class _SlowFakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            with construction_lock:
                constructions.append(model_name)
            # Widen the window an unsynchronised implementation would race in.
            time.sleep(0.05)

        def embed(self, texts):
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr(embeddings_module, "TextEmbedding", _SlowFakeTextEmbedding)
    model = EmbeddingModel()

    def _embed_once() -> None:
        barrier.wait()
        model.embed_one("concurrent first use")

    threads = [threading.Thread(target=_embed_once) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert constructions == [MODEL_NAME]
