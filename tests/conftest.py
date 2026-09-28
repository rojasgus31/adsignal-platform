import os
import platform
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest
import sklearn
from fastapi.testclient import TestClient

from adsignal.api.app import create_app, install_request_runtime
from adsignal.config import Settings
from adsignal.generator.anomalies import AnomalyInjector
from adsignal.generator.catalog import build_catalog
from adsignal.generator.events import generate_events
from adsignal.generator.models import ANOMALY_CAUSES, Catalog
from adsignal.ml.embeddings import EmbeddingModel
from adsignal.ml.evaluation import DetectorMetrics, TriageMetrics
from adsignal.ml.features import build_features
from adsignal.ml.registry import dumps_models, load_models, save_models
from adsignal.ml.training import BASELINE_WINDOW, train_and_evaluate
from adsignal.warehouse.artifact_store import ensure_artifact_schema, write_artifacts
from adsignal.warehouse.duckdb_io import open_duckdb, register_sources, run_dbt
from adsignal.warehouse.loader import ensure_schema as ensure_mart_schema
from adsignal.warehouse.loader import load_dim_campaign, load_mart_to_postgres
from adsignal.warehouse.metrics_store import (
    ensure_ml_schema,
    write_model_metrics,
    write_predictions,
)
from adsignal.warehouse.narrative_store import ensure_narrative_schema
from adsignal.warehouse.reader import read_campaign_daily
from adsignal.warehouse.schema import reset_schema_memo
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

# A third, separate database for `dropped_tables_client`: a genuinely fresh
# deployment (a serving database recreated for pgvector, say) has never had
# any `ensure_*_schema` call run against it at all, so no table exists -
# not "table exists, zero rows" like `EMPTY_TEST_DSN`/`empty_client`, but
# "relation does not exist". A prior run of this suite may have left tables
# behind here, so the fixture itself drops them every time rather than
# relying on this database staying pristine between runs.
DROPPED_TEST_DSN = os.environ.get(
    "ADSIGNAL_DROPPED_TEST_DSN",
    "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_dropped",
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


# A fourth, separate database for `test_schema_lock`: those tests drop every
# serving table and then hammer the read paths concurrently, so they cannot
# share a database with a fixture that expects tables to stay put.
CONCURRENCY_TEST_DSN = os.environ.get(
    "ADSIGNAL_CONCURRENCY_TEST_DSN",
    "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_concurrent",
)

# A sixth, separate database for `loaded_dimension` and everything in
# `test_dim_campaign.py`. That fixture calls `load_dim_campaign`, which
# TRUNCATEs before inserting, and it used to run against `TEST_DSN` - the same
# database the module-scoped `_api_seed` loads `dim_campaign` into. One
# function-scoped fixture replacing a module-scoped fixture's rows mid-session
# was harmless only for as long as both happened to build from the same
# `_build_small_dataset` call; changing either one's seed, advertiser count or
# date window would have turned it into an order-dependent failure in
# `test_campaigns_carry_human_names` that reads exactly like a serving-layer
# bug (Task 9b review, Finding 4). A database of its own removes the
# collision rather than documenting it.
DIM_TEST_DSN = os.environ.get(
    "ADSIGNAL_DIM_TEST_DSN",
    "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_dim",
)

# A fifth, separate database for `db_only_client`: the production topology
# Task 6b exists to make true, where the API and the training job do not
# share a filesystem. `_db_only_seed` never calls `ml.registry.save_models`,
# so nothing under this DSN's fixture could accidentally make `/score` pass
# by falling back to a directory - a shared database is the only thing this
# client is given.
DB_ONLY_TEST_DSN = os.environ.get(
    "ADSIGNAL_DB_ONLY_TEST_DSN",
    "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_db_only",
)


_ensure_test_database(TEST_DSN)
_ensure_test_database(EMPTY_TEST_DSN)
_ensure_test_database(DROPPED_TEST_DSN)
_ensure_test_database(CONCURRENCY_TEST_DSN)
_ensure_test_database(DB_ONLY_TEST_DSN)
_ensure_test_database(DIM_TEST_DSN)


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
def built_duckdb_with_orphan_campaign(tmp_path: Path) -> Path:
    """Like `built_duckdb`, but one campaign's advertiser_id matches no row
    in `raw_advertisers`.

    `build_catalog` always creates a campaign alongside its advertiser in the
    same loop (`generator/catalog.py`), so this case never occurs naturally
    in a generated dataset. It is manufactured directly against the raw
    source table here, to exercise `dim_campaign`'s left join and its
    fallback advertiser name (Task 1 review, Finding 1).
    """
    catalog, events_path, labels_path = _build_small_dataset(tmp_path)

    duckdb_path = tmp_path / "orphan.duckdb"
    with open_duckdb(duckdb_path) as conn:
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)
        conn.execute(
            "INSERT INTO raw_campaigns VALUES "
            "('cmp_orphan', 'adv_missing', 'Orphan Campaign', 'search', ?, ?, 100.0)",
            [date(2024, 1, 1), date(2024, 6, 30)],
        )

    run_dbt(project_dir=_DBT_DIR, duckdb_path=duckdb_path, command="build")

    return duckdb_path


@pytest.fixture
def loaded_dimension(built_duckdb):
    """Load both `fct_campaign_daily` and `dim_campaign` into `DIM_TEST_DSN`.

    Returns the DSN, which callers must use rather than reaching for
    `TEST_DSN`: this fixture owns a database of its own so that its
    truncating loads cannot replace the rows `_api_seed` put in `TEST_DSN`
    (see `DIM_TEST_DSN` above).

    Both tables are loaded from the same `built_duckdb` file, so
    `test_every_fact_campaign_resolves_to_a_dimension_row` compares a fact
    table and a dimension built from one dataset instead of depending on
    whatever another module happened to leave behind.

    Function-scoped to match `built_duckdb`'s own scope — a module-scoped
    fixture cannot depend on a function-scoped one.
    """
    if not _postgres_available(DIM_TEST_DSN):
        pytest.skip("postgres unavailable")
    load_mart_to_postgres(built_duckdb, DIM_TEST_DSN)
    load_dim_campaign(built_duckdb, DIM_TEST_DSN)
    return DIM_TEST_DSN


_ORPHAN_CAMPAIGN_ID = "cmp_without_a_dimension_row"
_ORPHAN_ADVERTISER_ID = "adv_without_a_dimension_row"
# Far enough in the future that `/anomalies` (ordered `event_date DESC`)
# returns this row on its first page whatever else the seed wrote.
_ORPHAN_EVENT_DATE = date(2099, 1, 1)


@pytest.fixture
def fact_row_without_dimension(client):
    """Add a campaign that exists in the facts but not in `dim_campaign`.

    A pipeline can load `fct_campaign_daily` and `anomaly_predictions`
    without `dim_campaign` (nothing in the schema enforces a foreign key), so
    the serving layer has to answer for a campaign it has no names for. The
    rows are inserted directly rather than generated, because
    `build_catalog` always creates a campaign alongside its advertiser and so
    never produces this case on its own.

    Function-scoped and removed again in teardown: `TEST_DSN` is shared with
    the module-scoped `client` fixture, and
    `test_every_fact_campaign_resolves_to_a_dimension_row` asserts that no
    such row survives.
    """
    if not _postgres_available(TEST_DSN):
        pytest.skip("Postgres not reachable")

    with psycopg.connect(TEST_DSN) as conn:
        conn.execute(
            """
            INSERT INTO fct_campaign_daily
                (event_date, campaign_id, advertiser_id, channel, impressions,
                 clicks, conversions, spend, revenue, is_anomaly)
            VALUES (%s, %s, %s, 'search', 1000, 10, 1, 50.0, 100.0, TRUE)
            """,
            (_ORPHAN_EVENT_DATE, _ORPHAN_CAMPAIGN_ID, _ORPHAN_ADVERTISER_ID),
        )
        conn.execute(
            """
            INSERT INTO anomaly_predictions
                (event_date, campaign_id, run_id, cause, confidence, severity,
                 escalate, anomaly_score)
            VALUES (%s, %s, 'run_2', 'budget_pacing', 0.9, 3, TRUE, 0.8)
            """,
            (_ORPHAN_EVENT_DATE, _ORPHAN_CAMPAIGN_ID),
        )
        conn.commit()

    try:
        yield _ORPHAN_CAMPAIGN_ID, _ORPHAN_ADVERTISER_ID, _ORPHAN_EVENT_DATE
    finally:
        with psycopg.connect(TEST_DSN) as conn:
            conn.execute(
                "DELETE FROM fct_campaign_daily WHERE campaign_id = %s",
                (_ORPHAN_CAMPAIGN_ID,),
            )
            conn.execute(
                "DELETE FROM anomaly_predictions WHERE campaign_id = %s",
                (_ORPHAN_CAMPAIGN_ID,),
            )
            conn.commit()


def _flat_series(days: int = 40, spend: float = 100.0) -> pd.DataFrame:
    """A flat, well-behaved single-campaign series, mart-shaped."""
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


def _labelled_series(days: int = 120) -> pd.DataFrame:
    """A `_flat_series` where every 6th day is an anomaly, cycling all 5 causes."""
    frame = _flat_series(days)
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


@pytest.fixture
def flat_series():
    """The `_flat_series` builder, for tests that vary its length or spend."""
    return _flat_series


@pytest.fixture
def labelled_series():
    """The `_labelled_series` builder, for tests that vary its length."""
    return _labelled_series


def _batch_predictions(mart: pd.DataFrame, artifacts: Path) -> pd.DataFrame:
    """The nightly scoring recipe, as `pipeline.assets.anomaly_scores` runs it.

    Deliberately a second, independent transcription of that asset rather than
    a shared helper. The equivalence test it feeds asserts that the live
    `POST /score` verdict matches the stored batch row; if both sides came out
    of one function the test would be a tautology. Duplication is the point
    here - if the two paths ever diverge, this is what fails.
    """
    detector, classifier = load_models(artifacts)
    features = build_features(mart, baseline_window=BASELINE_WINDOW)

    flagged = features[detector.predict(features)]
    assert not flagged.empty, "the detector flagged nothing in the seeded mart"

    predictions = classifier.predict(flagged)
    predictions["event_date"] = flagged["event_date"].to_numpy()
    predictions["campaign_id"] = flagged["campaign_id"].to_numpy()
    predictions["anomaly_score"] = detector.score(flagged)
    return predictions


def _train_artifacts(mart: pd.DataFrame, directory: Path) -> Path:
    """Fit both models on `mart` and persist them, returning `directory`.

    One recipe (`train_and_evaluate`, the same one the pipeline asset and the
    CI gate use) over two different inputs: a synthetic series for the model
    layer's own tests, and the seeded mart for the API's.
    """
    result = train_and_evaluate(mart, seed=0)
    save_models(directory, result.detector, result.classifier)
    return directory


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory) -> Path:
    """A registry-shaped artifact directory holding a fitted detector and triage pair.

    Built from a synthetic labelled series rather than from Postgres: what
    `adsignal.ml.scoring` needs is two fitted estimators on disk, and making
    the ML layer's tests wait on a database would contradict the layering the
    same tests assert.
    """
    return _train_artifacts(_labelled_series(180), tmp_path_factory.mktemp("trained_artifacts"))


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
    # The dimension is loaded from the same DuckDB file as the mart, so every
    # fact campaign here has a dimension row and the API's name fields hold
    # real names. `load_dim_campaign` truncates first, so rows left behind by
    # an earlier run of this suite cannot leak into these assertions.
    load_dim_campaign(duckdb_path, TEST_DSN)

    ensure_ml_schema(TEST_DSN)
    with psycopg.connect(TEST_DSN) as conn:
        for table in ("model_metrics", "drift_reports", "anomaly_predictions"):
            conn.execute(f"TRUNCATE TABLE {table}")  # noqa: S608
        conn.commit()

    triage = TriageMetrics(
        accuracy=0.8,
        macro_f1=0.75,
        per_class_f1={"budget_pacing": 0.8},
        confusion={"budget_pacing": {"budget_pacing": 5}},
        n_samples=5,
    )
    detector = DetectorMetrics(
        precision=0.6, recall=0.9, f1=0.72, roc_auc=0.85, n_samples=100, n_flagged=15
    )
    write_model_metrics(
        TEST_DSN, "run_1", datetime(2024, 1, 1, tzinfo=UTC), triage, detector, "/mlruns/run_1"
    )
    # The newest run gets real artifacts, trained on the mart just loaded, so
    # `POST /score` can actually run a model here instead of answering 503.
    # `model_metrics.artifact_uri` still gets the filesystem path (kept for
    # `dagster dev` and the ml-layer's own disk-based fixtures), but
    # `api.model_loader` now reads from `model_artifacts` in Postgres - see
    # the `write_artifacts` call below - so that is the row that actually
    # has to exist for these tests to see a 200 instead of a 503.
    mart = read_campaign_daily(TEST_DSN)
    artifacts = _train_artifacts(mart, tmp_path / "artifacts")
    detector_model, classifier_model = load_models(artifacts)
    write_model_metrics(
        TEST_DSN, "run_2", datetime(2024, 6, 1, tzinfo=UTC), triage, detector, str(artifacts)
    )
    detector_bytes, triage_bytes = dumps_models(detector_model, classifier_model)
    write_artifacts(
        TEST_DSN,
        "run_2",
        detector_bytes,
        triage_bytes,
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
    )

    # Real batch predictions, not a fabricated row: these are what
    # `GET /anomalies` serves, and `test_a_scored_day_matches_the_stored_batch_row`
    # asserts that `POST /score` reproduces one of them exactly. A hand-written
    # row would make that test unfalsifiable.
    write_predictions(TEST_DSN, "run_2", _batch_predictions(mart, artifacts))

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

    return str(mart["campaign_id"].iloc[0])


@pytest.fixture(scope="module")
def _db_only_seed(tmp_path_factory) -> str:
    """Seed `DB_ONLY_TEST_DSN` with mart rows and model artifacts, and nothing on disk.

    This is the production topology Task 6b exists to make true: `POST
    /score` served entirely from Postgres, with no artifact directory ever
    created for this run. `ml.registry.save_models` - the function that
    writes a directory - is never called here; the fitted models go straight
    from `train_and_evaluate` to `dumps_models` to `write_artifacts`, so
    there is no directory this fixture could be accidentally depending on.
    """
    if not _postgres_available(DB_ONLY_TEST_DSN):
        pytest.skip("Postgres not reachable")

    tmp_path = tmp_path_factory.mktemp("db_only_seed")
    catalog, events_path, labels_path = _build_small_dataset(tmp_path)
    duckdb_path = tmp_path / "db_only.duckdb"
    with open_duckdb(duckdb_path) as conn:
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)
    run_dbt(project_dir=_DBT_DIR, duckdb_path=duckdb_path, command="build")

    ensure_mart_schema(DB_ONLY_TEST_DSN)
    with psycopg.connect(DB_ONLY_TEST_DSN) as conn:
        conn.execute("TRUNCATE TABLE fct_campaign_daily")
        conn.commit()
    load_mart_to_postgres(duckdb_path, DB_ONLY_TEST_DSN)

    ensure_artifact_schema(DB_ONLY_TEST_DSN)
    with psycopg.connect(DB_ONLY_TEST_DSN) as conn:
        conn.execute("TRUNCATE TABLE model_artifacts")
        conn.commit()

    mart = read_campaign_daily(DB_ONLY_TEST_DSN)
    result = train_and_evaluate(mart, seed=0)
    detector_bytes, triage_bytes = dumps_models(result.detector, result.classifier)
    write_artifacts(
        DB_ONLY_TEST_DSN,
        "db_only_run",
        detector_bytes,
        triage_bytes,
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
    )

    return str(mart["campaign_id"].iloc[0])


@pytest.fixture(scope="module")
def db_only_client(_db_only_seed: str):
    """A `TestClient` that can only ever have scored from `model_artifacts`.

    No filesystem path was written for this run at all - see `_db_only_seed`
    - so a 200 from this client's `/score` is proof the database is a
    sufficient serving source on its own, not just a fallback the loader
    happens to prefer.
    """
    app = create_app(Settings(postgres_dsn=DB_ONLY_TEST_DSN))
    with TestClient(app) as test_client:
        yield test_client


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
    ensure_artifact_schema(EMPTY_TEST_DSN)
    with psycopg.connect(EMPTY_TEST_DSN) as conn:
        for table in (
            "fct_campaign_daily",
            "dim_campaign",
            "model_metrics",
            "drift_reports",
            "anomaly_predictions",
            "entity_embeddings",
            "narrative_runs",
            "model_artifacts",
        ):
            conn.execute(f"TRUNCATE TABLE {table}")  # noqa: S608
        conn.commit()

    app = create_app(Settings(postgres_dsn=EMPTY_TEST_DSN))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def dropped_tables_client():
    """A `TestClient` against a database where none of the serving tables exist.

    This is the state `empty_client` cannot cover: `empty_client` runs every
    `ensure_*_schema` before truncating, so it only ever exercises
    empty-but-existing tables. A genuinely fresh deployment - a serving
    database recreated for pgvector, say, before the pipeline has ever run
    against it - has no tables at all, and `DROP TABLE` (not `TRUNCATE`) is
    what reproduces that. No `ensure_*_schema` call is made here; the read
    path itself is responsible for tolerating a missing table.
    """
    if not _postgres_available(DROPPED_TEST_DSN):
        pytest.skip("Postgres not reachable")

    # `apply_schema` now memoizes "already ensured" per (dsn, statements) in
    # this process (see `warehouse/schema.py`), so dropping tables below must
    # be paired with forgetting the memo - otherwise the read path's
    # `ensure_*_schema` calls would skip DDL they need to actually rerun.
    reset_schema_memo(DROPPED_TEST_DSN)

    with psycopg.connect(DROPPED_TEST_DSN) as conn:
        for table in (
            "fct_campaign_daily",
            "dim_campaign",
            "model_metrics",
            "drift_reports",
            "anomaly_predictions",
            "entity_embeddings",
            "narrative_runs",
            "model_artifacts",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")  # noqa: S608
        conn.commit()

    app = create_app(Settings(postgres_dsn=DROPPED_TEST_DSN))
    with TestClient(app) as test_client:
        yield test_client


# Every fixture that hands a test a `TestClient`. `reset_ask_state` below
# only materializes the ones the test already asked for.
_CLIENT_FIXTURES = ("client", "empty_client", "dropped_tables_client", "db_only_client")


@pytest.fixture(autouse=True)
def reset_request_runtime(request):
    """Give every test fresh rate-limit budgets and an empty `/ask` cache.

    The client fixtures are module-scoped, so without this the `/ask` tests
    are order-coupled: `test_ask_is_rate_limited_per_ip` drains the bucket to
    zero and nothing refills it deterministically, so any `/ask` test that
    happens to run after it would see a 429 instead of its own answer. This
    resets both collaborators on `app.state` before each test, which is only
    possible because they are injected rather than module-level globals.
    """
    for name in _CLIENT_FIXTURES:
        if name in request.fixturenames:
            install_request_runtime(request.getfixturevalue(name).app)
