"""Review Focus 3: the dashboard is on a different origin than the API."""

import os

from fastapi.testclient import TestClient

from adsignal.api.app import create_app
from adsignal.api.cors import cors_kwargs
from adsignal.config import Settings

ALLOWED = "http://localhost:5173"
DENIED = "https://not-my-site.example"


def _client(origins: str) -> TestClient:
    os.environ["ADSIGNAL_CORS_ORIGINS"] = origins
    try:
        return TestClient(create_app())
    finally:
        del os.environ["ADSIGNAL_CORS_ORIGINS"]


def test_a_preflight_from_an_allowed_origin_is_accepted() -> None:
    response = _client(ALLOWED).options(
        "/search",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED


def test_a_get_from_an_allowed_origin_carries_the_header() -> None:
    response = _client(ALLOWED).get("/health", headers={"Origin": ALLOWED})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED


def test_an_origin_not_on_the_list_gets_no_allow_header() -> None:
    """The browser, not the server, enforces this — the absent header is the denial."""
    response = _client(ALLOWED).get("/health", headers={"Origin": DENIED})
    assert "access-control-allow-origin" not in response.headers


def test_several_origins_are_read_from_one_comma_separated_value() -> None:
    settings = Settings(cors_origins=("https://a.example", "https://b.example"))
    assert cors_kwargs(settings)["allow_origins"] == [
        "https://a.example",
        "https://b.example",
    ]


def test_credentials_are_never_allowed() -> None:
    """The API has no cookies or auth, so allowing credentials would only widen
    the blast radius of a mistake in the origin list."""
    assert cors_kwargs(Settings(cors_origins=(ALLOWED,)))["allow_credentials"] is False


def test_a_wildcard_origin_is_rejected_at_construction() -> None:
    """`*` on a public API is a configuration accident, not a choice."""
    import pytest

    with pytest.raises(ValueError, match="wildcard"):
        Settings(cors_origins=("*",))
