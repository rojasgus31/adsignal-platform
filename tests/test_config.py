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
