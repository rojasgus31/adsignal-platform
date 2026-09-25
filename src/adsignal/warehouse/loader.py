"""Loading aggregated marts from DuckDB into Postgres.

This is the only module that talks to Postgres. Loads are full refreshes:
each load truncates the serving table and reinserts every mart row inside a
single transaction. The mart is a full-refresh dbt `table` and is small, so
this is both simpler than a per-key upsert and a stronger guarantee — a
per-key upsert (`INSERT ... ON CONFLICT DO UPDATE`) is idempotent for keys
that reappear, but rows from a previous run that a new run does not produce
(e.g. after changing `ADSIGNAL_TARGET_EVENTS`, the seed, or the date window)
would otherwise survive forever, silently mixing two datasets in the
serving table.
"""

from pathlib import Path

from psycopg import sql

from adsignal.warehouse.duckdb_io import open_duckdb
from adsignal.warehouse.pool import connection
from adsignal.warehouse.schema import apply_schema

_MART_COLUMNS = (
    "event_date",
    "campaign_id",
    "advertiser_id",
    "channel",
    "impressions",
    "clicks",
    "conversions",
    "spend",
    "revenue",
    "ctr",
    "cpc",
    "cpm",
    "cpa",
    "roas",
    "anomaly_cause",
    "anomaly_severity",
    "is_anomaly",
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS fct_campaign_daily (
    event_date       DATE          NOT NULL,
    campaign_id      TEXT          NOT NULL,
    advertiser_id    TEXT          NOT NULL,
    channel          TEXT          NOT NULL,
    impressions      BIGINT        NOT NULL,
    clicks           BIGINT        NOT NULL,
    conversions      BIGINT        NOT NULL,
    spend            DOUBLE PRECISION NOT NULL,
    revenue          DOUBLE PRECISION NOT NULL,
    ctr              DOUBLE PRECISION,
    cpc              DOUBLE PRECISION,
    cpm              DOUBLE PRECISION,
    cpa              DOUBLE PRECISION,
    roas             DOUBLE PRECISION,
    anomaly_cause    TEXT,
    anomaly_severity INTEGER,
    is_anomaly       BOOLEAN       NOT NULL,
    PRIMARY KEY (event_date, campaign_id)
);
CREATE INDEX IF NOT EXISTS idx_fcd_campaign ON fct_campaign_daily (campaign_id, event_date);
CREATE INDEX IF NOT EXISTS idx_fcd_anomaly ON fct_campaign_daily (event_date) WHERE is_anomaly;

CREATE TABLE IF NOT EXISTS dim_campaign (
    campaign_id     TEXT NOT NULL PRIMARY KEY,
    campaign_name   TEXT NOT NULL,
    advertiser_id   TEXT NOT NULL,
    advertiser_name TEXT NOT NULL,
    channel         TEXT NOT NULL
);
"""

_DIM_CAMPAIGN_COLUMNS = (
    "campaign_id",
    "campaign_name",
    "advertiser_id",
    "advertiser_name",
    "channel",
)


def ensure_schema(dsn: str) -> None:
    """Create the serving table and its indexes if they do not exist.

    Serialised against every other `ensure_*_schema` helper by
    `apply_schema`; see that module for why concurrent `IF NOT EXISTS` DDL
    needs a lock.
    """
    apply_schema(dsn, _CREATE_TABLE)


def load_mart_to_postgres(duckdb_path: Path, dsn: str, batch_size: int = 10_000) -> int:
    """Full-refresh the DuckDB mart into Postgres. Returns the number of rows loaded.

    Truncates the serving table and reinserts every row inside a single
    transaction, so the final table always matches this run's mart exactly —
    including removing rows for keys the previous run produced but this one
    does not.
    """
    ensure_schema(dsn)

    columns = sql.SQL(", ").join(sql.Identifier(name) for name in _MART_COLUMNS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in _MART_COLUMNS)
    statement = sql.SQL(
        "INSERT INTO fct_campaign_daily ({columns}) VALUES ({placeholders})"
    ).format(columns=columns, placeholders=placeholders)

    total = 0
    select = f"SELECT {', '.join(_MART_COLUMNS)} FROM main_marts.fct_campaign_daily"  # noqa: S608

    with open_duckdb(duckdb_path) as duck, connection(dsn) as pg:
        cursor = duck.execute(select)
        with pg.cursor() as pg_cursor:
            pg_cursor.execute("TRUNCATE TABLE fct_campaign_daily")
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                pg_cursor.executemany(statement, batch)
                total += len(batch)
        pg.commit()

    return total


def load_dim_campaign(duckdb_path: Path, dsn: str, batch_size: int = 10_000) -> int:
    """Full-refresh `dim_campaign` into Postgres. Returns the number of rows loaded.

    Mirrors `load_mart_to_postgres` exactly: truncate then batched insert
    inside one transaction, committed once, so a cancelled run never leaves a
    half-truncated table for a reader to see.
    """
    ensure_schema(dsn)

    columns = sql.SQL(", ").join(sql.Identifier(name) for name in _DIM_CAMPAIGN_COLUMNS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in _DIM_CAMPAIGN_COLUMNS)
    statement = sql.SQL("INSERT INTO dim_campaign ({columns}) VALUES ({placeholders})").format(
        columns=columns, placeholders=placeholders
    )

    total = 0
    select = f"SELECT {', '.join(_DIM_CAMPAIGN_COLUMNS)} FROM main_marts.dim_campaign"  # noqa: S608

    with open_duckdb(duckdb_path) as duck, connection(dsn) as pg:
        cursor = duck.execute(select)
        with pg.cursor() as pg_cursor:
            pg_cursor.execute("TRUNCATE TABLE dim_campaign")
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                pg_cursor.executemany(statement, batch)
                total += len(batch)
        pg.commit()

    return total
