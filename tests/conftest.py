import os
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from adsignal.api.app import create_app
from adsignal.config import Settings
from adsignal.generator.anomalies import AnomalyInjector
from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events
from adsignal.generator.models import ANOMALY_CAUSES, Catalog
from adsignal.ml.embeddings import EmbeddingModel
from adsignal.ml.evaluation import DetectorMetrics, TriageMetrics
from adsignal.warehouse.duckdb_io import open_duckdb, register_sources, run_dbt
from adsignal.warehouse.loader import ensure_schema as ensure_mart_schema
from adsignal.warehouse.loader import load_mart_to_postgres
from adsignal.warehouse.metrics_store import (
    ensure_ml_schema,
    write_model_metrics,
    write_predictions,
)
from adsignal.warehouse.narrative_store import ensure_narrative_schema
from adsignal.warehouse.vector_store import EmbeddingRow, ensure_vector_schema, upsert_embeddings
from adsignal.warehouse.writer import write_events_parquet, write_labels_parquet

_DBT_DIR = Path(__file__).resolve().parents[1] / "dbt"

# The database the demo/dashboard reads. Tests must never point at it - see
# `_ensure_test_database` below (I2 in the ML layer review).
_SERVING_DSN = "postgresql://adsignal:adsignal@localhost:5432/adsignal"

# Every DSN-driven test module falls back to this constant, so the default is
# a sibling database on the same server rather than the serving one. Set
# ADSIGNAL_TEST_DSN to point elsewhere (a CI service, a different host).
TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)

# A second, separate database for `empty_client`: it must point at a database
# whose tables exist but are empty, and `TEST_DSN` is shared with (and
# populated by) the `client`/`seeded_campaign_id` fixtures below, so it cannot
# double as the empty one without the two fixtures fighting over the same
# rows.
EMPTY_TEST_DSN = os.environ.get(
    "ADSIGNAL_EMPTY_TEST_DSN",
    "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_empty",
)


def _ensure_test_database(dsn: str) -> None:
    """Create the database `dsn` names if it does not already exist.

    The suite truncates `model_metrics`/`drift_reports`/`anomaly_predictions`
    and does a full refresh of `fct_campaign_daily` (see `test_metrics_store`,
    `test_ml_assets`, `test_loader`). Running that against the same database
    the dashboard reads means "run the tests" and "demo the dashboard" are
    mutually destructive - which is exactly what happened during the final
    review (I2). This runs at import time, before any test module's own
    `_postgres_available()` skip guard evaluates its DSN, so the database
    exists by the time collection needs it.
    """
    if dsn == _SERVING_DSN:
        raise RuntimeError(
            "ADSIGNAL_TEST_DSN resolved to the serving database "
            f"({_SERVING_DSN}); running the suite would truncate model_metrics, "
            "drift_reports and anomaly_predictions and replace fct_campaign_daily "
            "wholesale under a running demo. Point ADSIGNAL_TEST_DSN at a "
            "separate database instead."
        )
    info = psycopg.conninfo.conninfo_to_dict(dsn)
    dbname = info.pop("dbname", None) or "adsignal_test"
    admin_dsn = psycopg.conninfo.make_conninfo(**info, dbname="postgres")
    try:
        with psycopg.connect(admin_dsn, connect_timeout=2, autocommit=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
            ).fetchone()
            if exists is None:
                conn.execute(f'CREATE DATABASE "{dbname}"')
    except psycopg.OperationalError:
        # Postgres itself isn't reachable here; each test module's own
        # `_postgres_available()` guard is what skips in that case.
        return


_ensure_test_database(TEST_DSN)
_ensure_test_database(EMPTY_TEST_DSN)


def _postgres_available(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_dbt_packages() -> None:
    """Install dbt packages once per session if they are missing.

    `dbt/dbt_packages/` is gitignored, so a fresh clone that runs the test
    suite without first running `dbt deps` (README Quick Start, or CI/the
    Dockerfile) would otherwise fail every dbt-building test. This makes the
    suite self-sufficient in that third environment too.
    """
    if (_DBT_DIR / "dbt_packages").exists():
        return
    subprocess.run(
        ["dbt", "deps", "--project-dir", str(_DBT_DIR), "--profiles-dir", str(_DBT_DIR)],
        check=True,
    )


def _build_small_dataset(tmp_path: Path) -> tuple[Catalog, Path, Path]:
    """Generate a small labelled dataset on disk.

    Shared by `small_dataset` and `built_duckdb` so the shape of the demo
    dataset (seed, advertiser/campaign/creative counts, date window) is
    defined in exactly one place - two independent copies would let the
    fixtures silently drift apart while both kept passing.
    """
    catalog = build_catalog(
        seed=99,
        n_advertisers=3,
        campaigns_per_advertiser=3,
        creatives_per_campaign=2,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )
    injector = AnomalyInjector(seed=99, rate=0.08)
    events_path = tmp_path / "raw_events.parquet"
    labels_path = tmp_path / "anomaly_labels.parquet"

    write_events_parquet(injector.rows(generate_events(catalog, seed=99)), events_path)
    write_labels_parquet(injector.labels, labels_path)

    return catalog, events_path, labels_path


@pytest.fixture
def small_dataset(tmp_path: Path):
    """Generate a small labelled dataset on disk and return its paths."""
    return _build_small_dataset(tmp_path)


@pytest.fixture
def built_duckdb(tmp_path: Path) -> Path:
    """Generate a small dataset, load it into DuckDB, and run dbt on it.

    Returns the path to a DuckDB file whose `main_staging` models exist, for
    tests that read from the staging layer rather than from Postgres.
    """
    catalog, events_path, labels_path = _build_small_dataset(tmp_path)

    duckdb_path = tmp_path / "built.duckdb"
    with open_duckdb(duckdb_path) as conn:
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)

    run_dbt(project_dir=_DBT_DIR, duckdb_path=duckdb_path, command="build")

    return duckdb_path


@pytest.fixture
def flat_series():
    """Build a flat, well-behaved single-campaign series."""

    def _build(days: int = 40, spend: float = 100.0) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=days, freq="D")
        return pd.DataFrame(
            {
                "event_date": dates,
                "campaign_id": ["cmp_a"] * days,
                "advertiser_id": ["adv_a"] * days,
                "channel": ["search"] * days,
                "impressions": [1000] * days,
                "clicks": [20] * days,
                "conversions": [2] * days,
                "spend": [spend] * days,
                "revenue": [200.0] * days,
                "ctr": [0.02] * days,
                "cpc": [spend / 20] * days,
                "cpm": [spend] * days,
                "cpa": [spend / 2] * days,
                "roas": [200.0 / spend] * days,
                "anomaly_cause": [None] * days,
                "anomaly_severity": [np.nan] * days,
                "is_anomaly": [False] * days,
            }
        )

    return _build


@pytest.fixture
def labelled_series(flat_series):
    """A `flat_series` where every 6th day is an anomaly, cycling all 5 causes."""

    def _build(days: int = 120) -> pd.DataFrame:
        frame = flat_series(days)
        rng = np.random.default_rng(0)
        for index in range(5, days, 6):
            cause = ANOMALY_CAUSES[(index // 6) % len(ANOMALY_CAUSES)]
            frame.loc[index, "anomaly_cause"] = cause
            frame.loc[index, "anomaly_severity"] = 1 + (index // 6) % 5
            frame.loc[index, "is_anomaly"] = True
            multiplier = {
                "budget_pacing": 2.5,
                "creative_fatigue": 1.0,
                "audience_saturation": 1.3,
                "tracking_error": 1.0,
                "competitive_pressure": 1.6,
            }[cause]
            frame.loc[index, "spend"] = frame.loc[index, "spend"] * multiplier
            frame.loc[index, "clicks"] = int(
                frame.loc[index, "clicks"] * (0.45 if cause == "creative_fatigue" else 1.0)
            )
            frame.loc[index, "conversions"] = 0 if cause == "tracking_error" else 2
            frame.loc[index, "ctr"] = frame.loc[index, "clicks"] / frame.loc[index, "impressions"]
            frame.loc[index, "spend"] += rng.normal(0, 0.5)
        return frame

    return _build


@pytest.fixture(scope="module")
def _api_seed(tmp_path_factory) -> str:
    """Build a small dataset and load it into `TEST_DSN` once per test module.

    Populates the mart, two `model_metrics` rows (so the newest-first
    ordering test is meaningful), one anomaly prediction, and a few
    embeddings for `/search`. Returns a campaign_id known to be in the mart.
    """
    if not _postgres_available(TEST_DSN):
        pytest.skip("Postgres not reachable")

    tmp_path = tmp_path_factory.mktemp("api_seed")
    catalog, events_path, labels_path = _build_small_dataset(tmp_path)
    duckdb_path = tmp_path / "api.duckdb"
    with open_duckdb(duckdb_path) as conn:
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)
    run_dbt(project_dir=_DBT_DIR, duckdb_path=duckdb_path, command="build")

    ensure_mart_schema(TEST_DSN)
    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("TRUNCATE TABLE fct_campaign_daily")
        conn.commit()
    load_mart_to_postgres(duckdb_path, TEST_DSN)

    ensure_ml_schema(TEST_DSN)
    with psycopg.connect(TEST_DSN) as conn:
        for table in ("model_metrics", "drift_reports", "anomaly_predictions"):
            conn.execute(f"TRUNCATE TABLE {table}")  # noqa: S608
        conn.commit()

    triage = TriageMetrics(
        accuracy=0.8,
        macro_f1=0.75,
        per_class_f1={"budget_pacing": 0.8},
        confusion={"budget_pacing": {"budget_pacing": 4}},
        n_samples=5,
    )
    detector = DetectorMetrics(
        precision=0.6, recall=0.9, f1=0.72, roc_auc=0.85, n_samples=100, n_flagged=15
    )
    write_model_metrics(
        TEST_DSN, "run_1", datetime(2024, 1, 1, tzinfo=UTC), triage, detector, "/mlruns/run_1"
    )
    write_model_metrics(
        TEST_DSN, "run_2", datetime(2024, 6, 1, tzinfo=UTC), triage, detector, "/mlruns/run_2"
    )

    with psycopg.connect(TEST_DSN) as conn:
        campaign_id, event_date = conn.execute(
            "SELECT campaign_id, event_date FROM fct_campaign_daily "
            "ORDER BY campaign_id, event_date LIMIT 1"
        ).fetchone()

    predictions = pd.DataFrame(
        [
            {
                "event_date": event_date,
                "campaign_id": campaign_id,
                "cause": "budget_pacing",
                "confidence": 0.9,
                "severity": 3,
                "escalate": True,
                "anomaly_score": 0.87,
            }
        ]
    )
    write_predictions(TEST_DSN, "run_2", predictions)

    ensure_vector_schema(TEST_DSN)
    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("TRUNCATE TABLE entity_embeddings")
        conn.commit()
    model = EmbeddingModel()
    texts = {
        "cmp_winter": "winter coat clearance discount",
        "cmp_audit": "quarterly financial audit compliance",
        "cmp_summer": "summer sandals promotion sale",
    }
    vectors = model.embed(list(texts.values()))
    upsert_embeddings(
        TEST_DSN,
        [
            EmbeddingRow("campaign", entity_id, text, vector)
            for (entity_id, text), vector in zip(texts.items(), vectors, strict=True)
        ],
    )

    return str(campaign_id)


@pytest.fixture(scope="module")
def seeded_campaign_id(_api_seed: str) -> str:
    """A campaign_id known to exist in the seeded mart."""
    return _api_seed


@pytest.fixture(scope="module")
def client(_api_seed: str):
    """A `TestClient` against a `create_app` pointed at the seeded `TEST_DSN`."""
    app = create_app(Settings(postgres_dsn=TEST_DSN))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def empty_client():
    """A `TestClient` against a database whose tables exist but hold no rows.

    A separate database (`EMPTY_TEST_DSN`), not `TEST_DSN`: the `client`
    fixture above seeds `TEST_DSN` with mart rows, model metrics and
    embeddings, so reusing it here — even truncated — would race the two
    fixtures' setup against each other whenever both are used in the same
    session. `ensure_*` calls create every table without inserting any rows.
    """
    if not _postgres_available(EMPTY_TEST_DSN):
        pytest.skip("Postgres not reachable")

    ensure_mart_schema(EMPTY_TEST_DSN)
    ensure_ml_schema(EMPTY_TEST_DSN)
    ensure_vector_schema(EMPTY_TEST_DSN)
    ensure_narrative_schema(EMPTY_TEST_DSN)
    with psycopg.connect(EMPTY_TEST_DSN) as conn:
        for table in (
            "fct_campaign_daily",
            "model_metrics",
            "drift_reports",
            "anomaly_predictions",
            "entity_embeddings",
            "narrative_runs",
        ):
            conn.execute(f"TRUNCATE TABLE {table}")  # noqa: S608
        conn.commit()

    app = create_app(Settings(postgres_dsn=EMPTY_TEST_DSN))
    with TestClient(app) as test_client:
        yield test_client
