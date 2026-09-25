"""The FastAPI application factory.

`create_app` is the api layer's single entry point: it wires one shared
`Settings` and one shared `EmbeddingModel` into `app.state`, then mounts
every router. Nothing here talks to Postgres directly — that stays inside
`warehouse/`.
"""

from fastapi import FastAPI

from adsignal.api.routes import anomalies, campaigns, health, models, search
from adsignal.config import Settings
from adsignal.ml.embeddings import EmbeddingModel

API_TITLE = "AdSignal Serving API"
API_DESCRIPTION = (
    "Read-only access to the AdSignal campaign mart, anomaly predictions, "
    "model evaluation history, and semantic search over stored entities."
)
API_VERSION = "0.1.0"


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

    app.include_router(health.router)
    app.include_router(campaigns.router)
    app.include_router(anomalies.router)
    app.include_router(models.router)
    app.include_router(search.router)

    return app
