"""Loading aggregated marts from DuckDB into Postgres.

This is the only module that talks to Postgres. Loads are upserts keyed on
(event_date, campaign_id), so re-running a pipeline never duplicates rows.
"""

from pathlib import Path

import psycopg
from psycopg import sql

from adsignal.warehouse.duckdb_io import open_duckdb

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
"""


def ensure_schema(dsn: str) -> None:
    """Create the serving table and its indexes if they do not exist."""
    with psycopg.connect(dsn) as conn:
        conn.execute(_CREATE_TABLE)
        conn.commit()


def load_mart_to_postgres(duckdb_path: Path, dsn: str, batch_size: int = 10_000) -> int:
    """Upsert the DuckDB mart into Postgres. Returns the number of rows upserted."""
    ensure_schema(dsn)

    columns = sql.SQL(", ").join(sql.Identifier(name) for name in _MART_COLUMNS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in _MART_COLUMNS)
    updates = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(name))
        for name in _MART_COLUMNS
        if name not in ("event_date", "campaign_id")
    )
    statement = sql.SQL(
        "INSERT INTO fct_campaign_daily ({columns}) VALUES ({placeholders}) "
        "ON CONFLICT (event_date, campaign_id) DO UPDATE SET {updates}"
    ).format(columns=columns, placeholders=placeholders, updates=updates)

    total = 0
    select = f"SELECT {', '.join(_MART_COLUMNS)} FROM main_marts.fct_campaign_daily"  # noqa: S608

    with open_duckdb(duckdb_path) as duck, psycopg.connect(dsn) as pg:
        cursor = duck.execute(select)
        with pg.cursor() as pg_cursor:
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                pg_cursor.executemany(statement, batch)
                total += len(batch)
        pg.commit()

    return total
