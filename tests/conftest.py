import os
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest

from adsignal.generator.anomalies import AnomalyInjector
from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events
from adsignal.generator.models import ANOMALY_CAUSES
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


@pytest.fixture
def small_dataset(tmp_path: Path):
    """Generate a small labelled dataset on disk and return its paths."""
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
