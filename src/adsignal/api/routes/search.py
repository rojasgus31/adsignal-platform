"""`POST /search` — semantic search over stored entity embeddings."""

from fastapi import APIRouter, Depends

from adsignal.api.deps import get_embedding_model, get_settings
from adsignal.api.schemas import SearchHitOut, SearchRequest, SearchResponse
from adsignal.config import Settings
from adsignal.ml.embeddings import EmbeddingModel
from adsignal.warehouse.semantic_search import semantic_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def post_search(
    request: SearchRequest,
    settings: Settings = Depends(get_settings),
    model: EmbeddingModel = Depends(get_embedding_model),
) -> SearchResponse:
    """Return the entities most similar in meaning to the query, nearest first."""
    hits = semantic_search(
        settings.postgres_dsn,
        request.query,
        model,
        limit=request.limit,
        kind=request.kind,
    )
    return SearchResponse(
        hits=[
            SearchHitOut(
                entity_kind=hit.entity_kind,
                entity_id=hit.entity_id,
                text=hit.text,
                distance=hit.distance,
            )
            for hit in hits
        ]
    )
