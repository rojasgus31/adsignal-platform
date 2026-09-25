"""Semantic search: embed a query, then find its nearest stored entities."""

from adsignal.ml.embeddings import EmbeddingModel
from adsignal.warehouse.vector_store import SearchHit, search_similar


def semantic_search(
    dsn: str,
    query: str,
    model: EmbeddingModel,
    limit: int = 10,
    kind: str | None = None,
) -> list[SearchHit]:
    """Return the entities most similar in meaning to `query`, nearest first.

    The model is passed in rather than constructed here so a caller — the API,
    which serves many requests — loads the weights once.
    """
    return search_similar(dsn, model.embed_one(query), limit=limit, kind=kind)
