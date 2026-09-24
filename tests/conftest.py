import subprocess
from datetime import date
from pathlib import Path

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
