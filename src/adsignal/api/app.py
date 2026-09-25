"""The FastAPI application factory.

`create_app` is the api layer's single entry point: it wires one shared
`Settings`, one shared `EmbeddingModel`, one shared `ScoringModelHolder` and
the shared runtime collaborators guarding `/search`, `/score` and `/ask` into
`app.state`, then mounts every router. Nothing here talks to Postgres
directly — that stays inside `warehouse/`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adsignal.api.cache import TtlCache
from adsignal.api.cors import cors_kwargs
from adsignal.api.model_loader import ScoringModelHolder
from adsignal.api.ratelimit import TokenBucket
from adsignal.api.routes import (
    anomalies,
    ask,
    campaigns,
    evidence,
    health,
    models,
    operations,
    score,
    search,
)
from adsignal.api.routes.ask import (
    ASK_CACHE_CAPACITY,
    ASK_CACHE_TTL_SECONDS,
    ASK_RATE_LIMIT_CAPACITY,
    ASK_RATE_LIMIT_REFILL_PER_SECOND,
)
from adsignal.api.routes.score import (
    SCORE_RATE_LIMIT_CAPACITY,
    SCORE_RATE_LIMIT_REFILL_PER_SECOND,
)
from adsignal.api.routes.search import (
    SEARCH_RATE_LIMIT_CAPACITY,
    SEARCH_RATE_LIMIT_REFILL_PER_SECOND,
)
from adsignal.config import Settings
from adsignal.ml.embeddings import EmbeddingModel
from adsignal.warehouse.pool import acquire_pool, release_pool

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

    `/ask`, `/search` and `/score` get separate buckets: they cost very
    different amounts per call, so they are tuned independently.
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
    app.state.score_rate_limiter = TokenBucket(
        capacity=SCORE_RATE_LIMIT_CAPACITY,
        refill_per_second=SCORE_RATE_LIMIT_REFILL_PER_SECOND,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Release this app's claim on its warehouse connection pool at shutdown.

    The pools in `adsignal.warehouse.pool` are process-wide, keyed by DSN, not
    per-`app`: two apps (or an API and a pipeline task) can share one DSN in
    the same process. `acquire_pool`/`release_pool` are refcounted per DSN so
    this app's shutdown only closes the pool once every other owner of the
    same DSN has also released it - closing every pool in the process
    regardless of ownership (the previous behaviour) would tear down a
    pool a sibling app, or a pipeline running in the same process, is still
    using. This still matters for the test suite building many short-lived
    apps in one process: each release now only affects its own DSN.
    """
    dsn = app.state.settings.postgres_dsn
    acquire_pool(dsn)
    yield
    release_pool(dsn)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app.

    A single `EmbeddingModel` and a single `ScoringModelHolder` are
    constructed here, once, and shared across every request via `app.state`
    (see `adsignal.api.deps`) — constructing a fresh one per request would
    reload the embedder's ~90 MB of weights on every `/search` call, and
    unpickle both fitted estimators on every `/score`. Both load what they
    wrap lazily on first use, so building them here costs nothing at startup
    and, in particular, `create_app` still touches no database.
    """
    resolved_settings = settings if settings is not None else Settings()

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=_lifespan,
    )
    app.state.settings = resolved_settings
    app.state.embedding_model = EmbeddingModel()
    app.state.scoring_models = ScoringModelHolder(resolved_settings)
    install_request_runtime(app)

    app.add_middleware(CORSMiddleware, **cors_kwargs(resolved_settings))  # type: ignore[arg-type]

    app.include_router(health.router)
    app.include_router(campaigns.router)
    app.include_router(anomalies.router)
    app.include_router(models.router)
    app.include_router(evidence.router)
    app.include_router(operations.router)
    app.include_router(search.router)
    app.include_router(score.router)
    app.include_router(ask.router)

    return app
