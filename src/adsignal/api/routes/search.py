"""`POST /search` — semantic search over stored entity embeddings.

This is the most expensive endpoint in the API: it runs ONNX embedding
inference on caller-supplied text on every call, where `/ask` only resolves a
name and runs one bounded indexed query. It is therefore rate-limited too,
per caller IP, with a tighter budget than `/ask`'s — the cheapest
denial-of-service lever against the demo should not be the endpoint that does
the most work per request.

There is no response cache here, unlike `/ask`: a cache keyed on free text
would have to hold a 384-float vector's worth of hits per distinct query, and
the rate limit already bounds the inference cost.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from adsignal.api.client_ip import client_key
from adsignal.api.deps import (
    get_embedding_model,
    get_search_rate_limiter,
    get_settings,
)
from adsignal.api.ratelimit import TokenBucket
from adsignal.api.schemas import SearchHitOut, SearchRequest, SearchResponse
from adsignal.config import Settings
from adsignal.ml.embeddings import EmbeddingModel
from adsignal.warehouse.semantic_search import semantic_search

router = APIRouter(tags=["search"])

# Tighter than `/ask`'s 10-burst / 5-per-second: one call here is an ONNX
# forward pass, not an indexed lookup. Read by `create_app`, which builds the
# bucket and puts it on `app.state`.
SEARCH_RATE_LIMIT_CAPACITY = 5
SEARCH_RATE_LIMIT_REFILL_PER_SECOND = 1.0


@router.post("/search", response_model=SearchResponse)
def post_search(
    body: SearchRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    model: EmbeddingModel = Depends(get_embedding_model),
    rate_limiter: TokenBucket = Depends(get_search_rate_limiter),
) -> SearchResponse:
    """Return the entities most similar in meaning to the query, nearest first."""
    if not rate_limiter.allow(client_key(request, settings.trusted_proxy_hops)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    hits = semantic_search(
        settings.postgres_dsn,
        body.query,
        model,
        limit=body.limit,
        kind=body.kind,
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
