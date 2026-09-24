"""DuckDB access and dbt invocation.

This is the only module that opens a DuckDB connection.
"""

import os
import subprocess
from pathlib import Path

import duckdb

from adsignal.generator.models import Catalog


def open_duckdb(path: Path) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the DuckDB database at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def register_sources(
    conn: duckdb.DuckDBPyConnection,
    *,
    events_path: Path,
    labels_path: Path,
    catalog: Catalog,
) -> None:
    """Materialize the raw_* tables. Safe to call repeatedly."""
    conn.execute(
        "CREATE OR REPLACE TABLE raw_events AS SELECT * FROM read_parquet(?)",
        [str(events_path)],
    )
    conn.execute(
        "CREATE OR REPLACE TABLE raw_anomaly_labels AS SELECT * FROM read_parquet(?)",
        [str(labels_path)],
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE raw_advertisers (
            id VARCHAR, name VARCHAR, vertical VARCHAR
        )
        """
    )
    conn.executemany(
        "INSERT INTO raw_advertisers VALUES (?, ?, ?)",
        [(a.id, a.name, a.vertical) for a in catalog.advertisers],
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE raw_campaigns (
            id VARCHAR, advertiser_id VARCHAR, name VARCHAR, channel VARCHAR,
            start_date DATE, end_date DATE, daily_budget DOUBLE
        )
        """
    )
    conn.executemany(
        "INSERT INTO raw_campaigns VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (c.id, c.advertiser_id, c.name, c.channel, c.start_date, c.end_date, c.daily_budget)
            for c in catalog.campaigns
        ],
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE raw_creatives (
            id VARCHAR, campaign_id VARCHAR, headline VARCHAR, format VARCHAR
        )
        """
    )
    conn.executemany(
        "INSERT INTO raw_creatives VALUES (?, ?, ?, ?)",
        [(cr.id, cr.campaign_id, cr.headline, cr.format) for cr in catalog.creatives],
    )


def run_dbt(*, project_dir: Path, duckdb_path: Path, command: str = "build") -> None:
    """Run a dbt command against `duckdb_path`. Raises on failure."""
    result = subprocess.run(
        [
            "dbt",
            command,
            "--project-dir",
            str(project_dir),
            "--profiles-dir",
            str(project_dir),
        ],
        env={**_dbt_env(duckdb_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dbt {command} failed:\n{result.stdout}\n{result.stderr}")


def _dbt_env(duckdb_path: Path) -> dict[str, str]:
    return {**os.environ, "ADSIGNAL_DUCKDB_PATH": str(duckdb_path)}
