import os
from pathlib import Path

import psycopg
import pytest

from adsignal.warehouse.duckdb_io import open_duckdb, register_sources, run_dbt
from adsignal.warehouse.loader import ensure_schema, load_mart_to_postgres

DBT_DIR = Path(__file__).resolve().parents[1] / "dbt"

DSN = os.environ.get("ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal")


def _postgres_available() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="Postgres not reachable; start it with docker compose up -d"
)


@pytest.fixture
def built_duckdb(tmp_path, small_dataset):
    catalog, events_path, labels_path = small_dataset
    duckdb_path = tmp_path / "marts.duckdb"
    with open_duckdb(duckdb_path) as conn:
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)
    run_dbt(project_dir=DBT_DIR, duckdb_path=duckdb_path, command="build")
    return duckdb_path


@pytest.fixture(autouse=True)
def clean_table():
    ensure_schema(DSN)
    with psycopg.connect(DSN) as conn:
        conn.execute("TRUNCATE TABLE fct_campaign_daily")
        conn.commit()
    yield


def test_load_copies_every_mart_row(built_duckdb) -> None:
    loaded = load_mart_to_postgres(built_duckdb, DSN)
    with psycopg.connect(DSN) as conn:
        count = conn.execute("SELECT count(*) FROM fct_campaign_daily").fetchone()[0]
    assert count == loaded > 0


def test_load_is_idempotent(built_duckdb) -> None:
    """Review Focus 2: assets are declared idempotent, so a second run must
    update in place rather than append duplicate rows."""
    first = load_mart_to_postgres(built_duckdb, DSN)
    second = load_mart_to_postgres(built_duckdb, DSN)
    assert first == second

    with psycopg.connect(DSN) as conn:
        total = conn.execute("SELECT count(*) FROM fct_campaign_daily").fetchone()[0]
        duplicates = conn.execute(
            """
            SELECT count(*) FROM (
                SELECT event_date, campaign_id FROM fct_campaign_daily
                GROUP BY 1, 2 HAVING count(*) > 1
            ) AS d
            """
        ).fetchone()[0]

    assert total == first
    assert duplicates == 0


def test_reload_updates_changed_values(built_duckdb) -> None:
    load_mart_to_postgres(built_duckdb, DSN)
    with psycopg.connect(DSN) as conn:
        conn.execute("UPDATE fct_campaign_daily SET impressions = -1")
        conn.commit()

    load_mart_to_postgres(built_duckdb, DSN)
    with psycopg.connect(DSN) as conn:
        stale = conn.execute(
            "SELECT count(*) FROM fct_campaign_daily WHERE impressions = -1"
        ).fetchone()[0]
    assert stale == 0


def test_null_rate_metrics_survive_the_load(built_duckdb) -> None:
    load_mart_to_postgres(built_duckdb, DSN)
    with psycopg.connect(DSN) as conn:
        bad = conn.execute(
            "SELECT count(*) FROM fct_campaign_daily WHERE conversions = 0 AND cpa IS NOT NULL"
        ).fetchone()[0]
    assert bad == 0
