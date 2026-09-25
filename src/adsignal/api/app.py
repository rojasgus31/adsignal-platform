"""The FastAPI application factory.

`create_app` is the api layer's single entry point: it wires one shared
`Settings`, one shared `EmbeddingModel` and the shared runtime collaborators
guarding `/search` and `/ask` into `app.state`, then mounts every router. Nothing here talks
to Postgres directly — that stays inside `warehouse/`.
"""

from typing import Any

from fastapi import FastAPI

from adsignal.api.cache import TtlCache
from adsignal.api.ratelimit import TokenBucket
from adsignal.api.routes import anomalies, ask, campaigns, health, models, search
from adsignal.api.routes.ask import (
    ASK_CACHE_CAPACITY,
    ASK_CACHE_TTL_SECONDS,
    ASK_RATE_LIMIT_CAPACITY,
    ASK_RATE_LIMIT_REFILL_PER_SECOND,
)
from adsignal.api.routes.search import (
    SEARCH_RATE_LIMIT_CAPACITY,
    SEARCH_RATE_LIMIT_REFILL_PER_SECOND,
)
from adsignal.config import Settings
from adsignal.ml.embeddings import EmbeddingModel

API_TITLE = "AdSignal Serving API"
API_DESCRIPTION = (
    "Read-only access to the AdSignal campaign mart, anomaly predictions, "
    "model evaluation history, and semantic search over stored entities."
)
API_VERSION = "0.1.0"


def install_request_runtime(app: FastAPI) -> None:
    """Give `app` fresh rate-limit budgets and an empty response cache.

    Separate from `create_app` so the test suite can reset them between tests
    without rebuilding the whole app (and reloading the embedding model). All
    three live on `app.state`, reached through `deps.py`, so two apps in one
    process never share a budget or serve each other's cached answers.

    `/search` and `/ask` get separate buckets: they cost very different
    amounts per call, so they are tuned independently.
    """
    app.state.ask_rate_limiter = TokenBucket(
        capacity=ASK_RATE_LIMIT_CAPACITY,
        refill_per_second=ASK_RATE_LIMIT_REFILL_PER_SECOND,
    )
    app.state.ask_cache = TtlCache[Any](
        capacity=ASK_CACHE_CAPACITY, ttl_seconds=ASK_CACHE_TTL_SECONDS
    )
    app.state.search_rate_limiter = TokenBucket(
        capacity=SEARCH_RATE_LIMIT_CAPACITY,
        refill_per_second=SEARCH_RATE_LIMIT_REFILL_PER_SECOND,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app.

    A single `EmbeddingModel` is constructed here, once, and shared across
    every request via `app.state` (see `adsignal.api.deps`) — constructing a
    fresh one per request would reload its ~90 MB of weights on every
    `/search` call. The model's own weights still load lazily on first use,
    so building it here costs nothing at startup.
    """
    resolved_settings = settings if settings is not None else Settings()

    app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION)
    app.state.settings = resolved_settings
    app.state.embedding_model = EmbeddingModel()
    install_request_runtime(app)

    app.include_router(health.router)
    app.include_router(campaigns.router)
    app.include_router(anomalies.router)
    app.include_router(models.router)
    app.include_router(search.router)
    app.include_router(ask.router)

    return app
