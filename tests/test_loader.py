import os
from datetime import date
from pathlib import Path

import psycopg
import pytest

from adsignal.generator.anomalies import AnomalyInjector
from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events
from adsignal.warehouse.duckdb_io import open_duckdb, register_sources, run_dbt
from adsignal.warehouse.loader import ensure_schema, load_mart_to_postgres
from adsignal.warehouse.writer import write_events_parquet, write_labels_parquet

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


def test_reload_with_fewer_keys_drops_stale_rows(tmp_path, small_dataset) -> None:
    """C5: a per-key upsert is idempotent for keys that reappear but never
    deletes keys that don't. Load mart A, then a mart B with strictly fewer
    keys (a subset of advertisers), and assert the final row count equals
    B's — not A's plus B's. `test_load_is_idempotent` reloads the *same*
    file, which is exactly the case where a missing-delete bug is invisible;
    this covers the case where it is not."""
    catalog, events_path, labels_path = small_dataset

    def _build(name: str, n_advertisers: int) -> Path:
        sub_catalog = build_catalog(
            seed=99,
            n_advertisers=n_advertisers,
            campaigns_per_advertiser=3,
            creatives_per_campaign=2,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        )
        injector = AnomalyInjector(seed=99, rate=0.08)
        sub_events_path = tmp_path / f"{name}_events.parquet"
        sub_labels_path = tmp_path / f"{name}_labels.parquet"
        write_events_parquet(injector.rows(generate_events(sub_catalog, seed=99)), sub_events_path)
        write_labels_parquet(injector.labels, sub_labels_path)

        duckdb_path = tmp_path / f"{name}.duckdb"
        with open_duckdb(duckdb_path) as conn:
            register_sources(
                conn, events_path=sub_events_path, labels_path=sub_labels_path, catalog=sub_catalog
            )
        run_dbt(project_dir=DBT_DIR, duckdb_path=duckdb_path, command="build")
        return duckdb_path

    mart_a = _build("mart_a", n_advertisers=3)
    mart_b = _build("mart_b", n_advertisers=1)

    load_mart_to_postgres(mart_a, DSN)
    loaded_b = load_mart_to_postgres(mart_b, DSN)

    with psycopg.connect(DSN) as conn:
        final_count = conn.execute("SELECT count(*) FROM fct_campaign_daily").fetchone()[0]

    assert final_count == loaded_b


def test_null_rate_metrics_survive_the_load(built_duckdb) -> None:
    load_mart_to_postgres(built_duckdb, DSN)
    with psycopg.connect(DSN) as conn:
        bad = conn.execute(
            "SELECT count(*) FROM fct_campaign_daily WHERE conversions = 0 AND cpa IS NOT NULL"
        ).fetchone()[0]
    assert bad == 0
