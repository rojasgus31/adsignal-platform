import os
from datetime import date
from pathlib import Path

import pytest

from adsignal.config import Settings


@pytest.fixture(autouse=True)
def _no_ambient_adsignal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate these tests from ambient ADSIGNAL_* environment variables.

    The README's documented workflow is `cp .env.example .env` followed by
    tuning that file, and `Settings` reads `env_file=".env"` by default. A
    unit test of *defaults* must not depend on the reader's shell or on
    whatever `.env` happens to contain.
    """
    for name in tuple(os.environ):
        if name.startswith("ADSIGNAL_"):
            monkeypatch.delenv(name, raising=False)


def test_settings_have_documented_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.target_events == 10_000_000
    assert settings.seed == 20260923
    assert settings.start_date == date(2024, 1, 1)
    assert settings.end_date == date(2025, 12, 31)
    assert settings.data_dir == Path("data")


def test_settings_read_env_with_prefix(monkeypatch) -> None:
    monkeypatch.setenv("ADSIGNAL_TARGET_EVENTS", "5000")
    monkeypatch.setenv("ADSIGNAL_SEED", "7")
    settings = Settings()
    assert settings.target_events == 5000
    assert settings.seed == 7


def test_settings_repr_never_contains_a_secret() -> None:
    """An unhandled exception, debug endpoint, or reload error page can all end
    up printing, dumping, or JSON-serialising a `Settings` instance. None of
    its credential fields may show up on any of those four surfaces — and
    masking must not come at the cost of `postgres_dsn` still being a usable
    connection string for the dozens of call sites that read it directly.
    """
    sentinel = "sk-super-secret-key-do-not-leak"
    dsn = f"postgresql://adsignal:{sentinel}@localhost:5432/adsignal"
    settings = Settings(
        _env_file=None,
        postgres_dsn=dsn,
        gemini_api_key=sentinel,
        groq_api_key=sentinel,
        jev_api_key=sentinel,
        anthropic_api_key=sentinel,
    )

    assert sentinel not in repr(settings)
    assert sentinel not in str(settings)
    assert sentinel not in str(settings.model_dump())
    assert sentinel not in settings.model_dump_json()

    # Masking model_dump()/model_dump_json() must not break attribute access:
    # every call site that connects to Postgres reads settings.postgres_dsn
    # directly, and it must still be the real, usable DSN.
    assert settings.postgres_dsn == dsn
