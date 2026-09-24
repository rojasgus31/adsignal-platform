from datetime import date
from pathlib import Path

from adsignal.config import Settings


def test_settings_have_documented_defaults() -> None:
    settings = Settings()
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
