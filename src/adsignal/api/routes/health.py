"""`GET /health` — a liveness check."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def get_health() -> dict[str, str]:
    """Report that the service is up.

    Returns status only — never `Settings` or any part of it. A health
    endpoint that dumps configuration is how a database credential leaks;
    see `config.Settings._serialize_postgres_dsn`.
    """
    return {"status": "ok"}
