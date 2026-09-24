import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adsignal.generator.anomalies import AnomalyInjector
from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events
from adsignal.warehouse.writer import write_events_parquet, write_labels_parquet

_DBT_DIR = Path(__file__).resolve().parents[1] / "dbt"


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
