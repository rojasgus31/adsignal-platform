"""Application settings.

`Settings` is the only module that reads configuration *from* the
environment. `adsignal.warehouse.duckdb_io` *passes* settings on to the dbt
subprocess's environment — it does not itself read configuration.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the AdSignal data platform."""

    model_config = SettingsConfigDict(env_prefix="ADSIGNAL_", env_file=".env", extra="ignore")

    target_events: int = 10_000_000
    seed: int = 20260923
    start_date: date = date(2024, 1, 1)
    end_date: date = date(2025, 12, 31)
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/adsignal.duckdb")
    # repr=False: this DSN carries a live credential. str stays (dozens of call
    # sites across warehouse/ and pipeline/ read it as a plain string), but it
    # must never be printed by a default dataclass-style repr or str.
    postgres_dsn: str = Field(
        default="postgresql://adsignal:adsignal@localhost:5432/adsignal", repr=False
    )
    # Override for the dbt project location when running inside a container.
    # Not used in this task; consumed by a later task.
    dbt_dir: Path | None = None
    mlflow_dir: Path = Path("mlruns")
    # SecretStr: only build_chain() reads these, and .get_secret_value() there
    # is the one place the raw key is ever extracted. SecretStr also blocks
    # accidental f-string interpolation, not just repr/str.
    gemini_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    jev_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    narrative_timeout_seconds: float = 20.0

    @property
    def raw_events_path(self) -> Path:
        """Parquet destination for generated raw events."""
        return self.data_dir / "raw_events.parquet"

    @property
    def anomaly_labels_path(self) -> Path:
        """Parquet destination for ground-truth anomaly labels."""
        return self.data_dir / "anomaly_labels.parquet"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
