"""FastAPI dependency wiring.

Every dependency reads from `request.app.state`, which `create_app` populates
once at construction time — the `EmbeddingModel`, so its ~90 MB of weights
load once per process rather than once per `/search` call, and the rate
limiters and response cache guarding the two endpoints that do live work, so
they are per-app collaborators a test can replace rather than import-time
module globals shared by every app in the process.
"""

from typing import Any

from fastapi import Request

from adsignal.api.cache import TtlCache
from adsignal.api.model_loader import ScoringModelHolder
from adsignal.api.ratelimit import TokenBucket
from adsignal.config import Settings
from adsignal.ml.embeddings import EmbeddingModel
from adsignal.narrative.sqlgen import SqlGenerator


def get_settings(request: Request) -> Settings:
    """Return the `Settings` this app instance was constructed with."""
    settings: Settings = request.app.state.settings
    return settings


def get_embedding_model(request: Request) -> EmbeddingModel:
    """Return the single `EmbeddingModel` shared across requests."""
    model: EmbeddingModel = request.app.state.embedding_model
    return model


def get_ask_rate_limiter(request: Request) -> TokenBucket:
    """Return the `TokenBucket` bounding `/ask` for this app instance."""
    bucket: TokenBucket = request.app.state.ask_rate_limiter
    return bucket


def get_ask_cache(request: Request) -> TtlCache[Any]:
    """Return the bounded response cache `/ask` answers from, for this app."""
    cache: TtlCache[Any] = request.app.state.ask_cache
    return cache


def get_sql_generator(request: Request) -> SqlGenerator | None:
    """Return this app's question-to-SQL generator, or `None` if none is configured.

    `None` is the state a deployment with no `ADSIGNAL_GEMINI_API_KEY` is in,
    and `/ask` reports it as such. It lives on `app.state` like the rate
    limiters so a test can replace it with a stub provider, and so the
    generator's circuit breaker is per-app rather than a module global shared
    by every app in the process.
    """
    generator: SqlGenerator | None = request.app.state.sql_generator
    return generator


def get_search_rate_limiter(request: Request) -> TokenBucket:
    """Return the `TokenBucket` bounding `/search` for this app instance.

    Separate from `/ask`'s bucket: the two endpoints cost very different
    amounts per call (ONNX inference versus one indexed query), so they are
    tuned independently and one must not be able to spend the other's budget.
    """
    bucket: TokenBucket = request.app.state.search_rate_limiter
    return bucket


def get_score_rate_limiter(request: Request) -> TokenBucket:
    """Return the `TokenBucket` bounding `/score` for this app instance.

    Its own bucket, for the same reason `/search` has one separate from
    `/ask`: a `/score` call is a warehouse read plus a forward pass through
    an isolation forest and two gradient-boosted ensembles, and the endpoint
    that does the most work per request must not be the cheapest one to
    exhaust.
    """
    bucket: TokenBucket = request.app.state.score_rate_limiter
    return bucket


def get_scoring_models(request: Request) -> ScoringModelHolder:
    """Return the holder for this app's one shared detector/triage pair."""
    holder: ScoringModelHolder = request.app.state.scoring_models
    return holder


def get_summary_rate_limiter(request: Request) -> TokenBucket:
    """Return the `TokenBucket` bounding campaign summaries for this app."""
    bucket: TokenBucket = request.app.state.summary_rate_limiter
    return bucket


def get_summary_cache(request: Request) -> TtlCache[Any]:
    """Return the bounded cache campaign summaries are served from."""
    cache: TtlCache[Any] = request.app.state.summary_cache
    return cache
