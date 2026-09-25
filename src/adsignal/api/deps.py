"""FastAPI dependency wiring.

Every dependency reads from `request.app.state`, which `create_app` populates
once at construction time — the `EmbeddingModel`, so its ~90 MB of weights
load once per process rather than once per `/search` call, and `/ask`'s rate
limiter and response cache, so they are per-app collaborators a test can
replace rather than import-time module globals shared by every app in the
process.
"""

from typing import Any

from fastapi import Request

from adsignal.api.cache import TtlCache
from adsignal.api.ratelimit import TokenBucket
from adsignal.config import Settings
from adsignal.ml.embeddings import EmbeddingModel


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
