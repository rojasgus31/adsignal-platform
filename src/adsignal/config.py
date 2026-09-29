"""Application settings.

`Settings` is the only module that reads configuration *from* the
environment. `adsignal.warehouse.duckdb_io` *passes* settings on to the dbt
subprocess's environment — it does not itself read configuration.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr, field_serializer, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _mask_dsn_password(dsn: str) -> str:
    """Replace a DSN's password with a fixed placeholder, keeping host and database visible.

    `repr=False` only stops `__repr__`/`__str__` from printing this field; it does
    nothing for `model_dump()` or `model_dump_json()`, which pydantic still
    serialises verbatim. This is the masking those two paths need. Falls back to
    redacting the whole value if the DSN can't be parsed as a URL, since a
    credential leak is worse than a lost debugging hint.
    """
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return "REDACTED"
    if parts.password is None:
        return dsn
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    userinfo = f"{parts.username}:REDACTED" if parts.username else ":REDACTED"
    masked = parts._replace(netloc=f"{userinfo}@{host}")
    return urlunsplit(masked)


# The Gemini model every provider in this project calls. A hardcoded model
# name is a dated assertion about someone else's product: this project's
# previous default, `gemini-2.0-flash`, was retired, after which every call
# returned HTTP 404 with a perfectly valid key — a failure that reads like a
# broken endpoint rather than a retired model.
#
# To re-derive the current list for a given key:
#   GET https://generativelanguage.googleapis.com/v1beta/models?key=...
# and take a non-preview entry whose supportedGenerationMethods include
# `generateContent`. `gemini-3.5-flash` is the base stable release of the 3.x
# flash line; the 3.6/3.7/3.8 point releases rotate faster.
#
# Declared here, once, and passed to the providers. Neither
# `GeminiProvider` nor `GeminiSqlProvider` carries a model default of its
# own, so this constant cannot fall out of step with a second copy.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


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
    # The DSN generated `/ask` SQL runs on, and the only one it may ever run
    # on. It names the `adsignal_ro` login role created by
    # `warehouse.ask_reads.ensure_readonly_role`: SELECT on seven tables,
    # ownership of nothing, `model_artifacts` and `entity_embeddings`
    # revoked. Empty means `/ask` is not configured, which that endpoint
    # reports as a plain 200 rather than failing. repr=False for the same
    # reason as `postgres_dsn`: it carries a live credential.
    postgres_readonly_dsn: str = Field(default="", repr=False)
    # `adsignal.warehouse.pool` reads these once, when it opens the first pool
    # for a DSN. Every API route is a sync `def`, so Starlette runs it in its
    # request threadpool (anyio's default capacity is 40); `pool_max_size=4`
    # against that meant a burst of concurrent requests queued behind a
    # handful of connections for up to the pool's 30s default timeout before
    # failing. `pool_max_waiting` bounds how many callers may queue at all -
    # past it, a new caller fails immediately instead of piling on to an
    # already-saturated queue - and `pool_timeout_seconds` shortens the wait
    # so an overloaded pool fails fast rather than slowly.
    pool_min_size: int = 1
    pool_max_size: int = 10
    pool_timeout_seconds: float = 5.0
    pool_max_waiting: int = 64
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
    gemini_model: str = DEFAULT_GEMINI_MODEL
    """Which Gemini model to call, overridable with ADSIGNAL_GEMINI_MODEL.

    A setting rather than a literal so the next retirement is an environment
    change on a running deployment, not a code edit and a rebuild.
    """
    narrative_timeout_seconds: float = 20.0
    # The chain's total deadline across every provider, distinct from the
    # per-provider timeout above: four providers each allowed 20 s can stall
    # the nightly job for 80 s or more without this. See `NarrativeChain`.
    narrative_budget_seconds: float = 45.0
    trusted_proxy_hops: int = Field(default=0, ge=0, le=4)
    """How many proxies sit in front of this API.

    0 means the socket peer is the caller and `X-Forwarded-For` is ignored —
    the correct value for local runs and for any deployment reached directly.
    On Render, which terminates TLS at its own edge, this is 1. Setting it
    higher than the real number lets a caller forge an address and evade the
    rate limiter, so it is bounded and defaults to the safe end.
    """
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ()
    """Exact origins allowed to call this API from a browser.

    Read from ADSIGNAL_CORS_ORIGINS as a comma-separated list. Empty means no
    browser may call the API, which is the correct default for a service whose
    only consumer is named explicitly at deploy time. `NoDecode` stops
    pydantic-settings from trying to JSON-parse the raw env value before our
    validator below splits it on commas.
    """

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept "https://a,https://b" from the environment as a tuple."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("cors_origins")
    @classmethod
    def _reject_wildcard(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """`*` on a public API is an accident. Name the origins."""
        if "*" in value:
            raise ValueError("cors_origins must not contain a wildcard; name each origin")
        return value

    @field_serializer("postgres_dsn", "postgres_readonly_dsn", when_used="always")
    def _serialize_postgres_dsn(self, value: str) -> str:
        """Mask the password when this settings object is dumped or dumped to JSON.

        Both DSNs go through this. Attribute access
        (`settings.postgres_dsn`, `settings.postgres_readonly_dsn`) is
        untouched and keeps
        returning the real, usable DSN for the dozens of call sites that
        connect with it — only `model_dump()`/`model_dump_json()` go through
        this.
        """
        return _mask_dsn_password(value)

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
