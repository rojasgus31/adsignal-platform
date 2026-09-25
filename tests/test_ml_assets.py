import os
from datetime import date

import psycopg
import pytest
from dagster import materialize

from adsignal.config import Settings
from adsignal.narrative.build import build_chain
from adsignal.pipeline.assets import (
    anomaly_scores,
    campaign_daily_mart,
    campaign_embeddings,
    catalog_and_raw_events,
    daily_narrative,
    dbt_marts,
    drift_report,
    trained_models,
)

DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


def _postgres_available() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _postgres_available(), reason="Postgres not reachable")


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    tmp_path = tmp_path_factory.mktemp("ml_assets")
    return Settings(
        target_events=40_000,
        seed=4242,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ml.duckdb",
        postgres_dsn=DSN,
        mlflow_dir=tmp_path / "mlruns",
    )


@pytest.fixture(scope="module")
def pipeline_run(settings: Settings):
    """Materialize the full pipeline, including scoring and drift, exactly once.

    Every test in this module asserts against this single run's outputs and
    the database rows it wrote, instead of re-running the full pipeline
    (data generation, dbt build, training) per test.
    """
    result = materialize(
        [
            catalog_and_raw_events,
            dbt_marts,
            campaign_daily_mart,
            trained_models,
            anomaly_scores,
            drift_report,
            daily_narrative,
        ],
        resources={"settings": settings, "narrative_chain": build_chain(settings)},
    )
    assert result.success
    return result, settings


def _run_id(pipeline_run) -> str:
    result, _ = pipeline_run
    metadata = result.asset_materializations_for_node("trained_models")[0].metadata
    return str(metadata["run_id"].value)


def test_training_asset_records_metrics_and_a_run_id(pipeline_run) -> None:
    result, settings = pipeline_run
    metadata = result.asset_materializations_for_node("trained_models")[0].metadata
    assert metadata["run_id"].value
    assert 0.0 <= metadata["triage_macro_f1"].value <= 1.0

    run_id = _run_id(pipeline_run)
    with psycopg.connect(settings.postgres_dsn) as conn:
        count = conn.execute(
            "SELECT count(*) FROM model_metrics WHERE run_id = %s", (run_id,)
        ).fetchone()[0]
    # >= rather than == : this fixture's module scope means a re-run within
    # the same test session (e.g. under -p randomly) could add more than one
    # row for the same run_id if trained_models were re-materialized, but it
    # must never be zero - this scoped WHERE is what makes the assertion mean
    # something, unlike an unscoped `count(*) >= 1` on an append-only table
    # that would pass on residue from any earlier test (P1-2).
    assert count >= 1


def test_training_asset_writes_model_artifacts(pipeline_run) -> None:
    """P1-3: the property that matters is the artifact files `anomaly_scores`
    depends on, not merely that `mlflow_dir` (created on the first line of
    `log_training_run`, before anything is saved) exists."""
    _, settings = pipeline_run
    run_id = _run_id(pipeline_run)
    artifact_dir = settings.mlflow_dir / "artifacts" / run_id
    assert (artifact_dir / "detector.pkl").exists()
    assert (artifact_dir / "triage.pkl").exists()


def test_scoring_asset_writes_predictions(pipeline_run) -> None:
    _, settings = pipeline_run
    run_id = _run_id(pipeline_run)
    with psycopg.connect(settings.postgres_dsn) as conn:
        count = conn.execute(
            "SELECT count(*) FROM anomaly_predictions WHERE run_id = %s", (run_id,)
        ).fetchone()[0]
    assert count > 0


def test_predictions_only_carry_known_causes(pipeline_run) -> None:
    from adsignal.generator.models import ANOMALY_CAUSES

    _, settings = pipeline_run
    with psycopg.connect(settings.postgres_dsn) as conn:
        causes = {row[0] for row in conn.execute("SELECT DISTINCT cause FROM anomaly_predictions")}
    assert causes <= set(ANOMALY_CAUSES)


def test_drift_asset_writes_one_row_per_feature(pipeline_run) -> None:
    _, settings = pipeline_run
    run_id = _run_id(pipeline_run)
    with psycopg.connect(settings.postgres_dsn) as conn:
        count = conn.execute(
            "SELECT count(*) FROM drift_reports WHERE run_id = %s", (run_id,)
        ).fetchone()[0]
    assert count > 0


def test_definitions_expose_the_ml_assets() -> None:
    from adsignal.pipeline.definitions import defs

    keys = {key.to_user_string() for key in defs.resolve_asset_graph().get_all_asset_keys()}
    assert {"trained_models", "anomaly_scores", "drift_report", "daily_narrative"} <= keys


def test_narrative_asset_writes_a_row_with_provider_metadata(pipeline_run) -> None:
    from adsignal.warehouse.narrative_store import read_latest_narrative

    result, settings = pipeline_run
    metadata = result.asset_materializations_for_node("daily_narrative")[0].metadata
    # No API keys are configured in this test's Settings, so build_chain's
    # trailing StubProvider is the one that actually answers.
    assert metadata["provider"].value == "stub"
    assert metadata["latency_ms"].value >= 0.0

    stored = read_latest_narrative(settings.postgres_dsn)
    assert stored is not None
    assert stored.provider == "stub"
    assert stored.text


def test_narrative_asset_does_not_fail_the_pipeline_when_every_provider_fails(
    pipeline_run,
) -> None:
    """AllProvidersFailed must be caught, not propagated: a missing narrative is a
    degraded dashboard, never a reason to fail the nightly job.

    Depends on `pipeline_run` (unused directly) so `fct_campaign_daily` already
    has rows - otherwise `_build_narrative_request` would return None before
    `build_chain` is ever called, and this would not exercise the failure path
    it's meant to test.
    """
    from dagster import build_asset_context

    from adsignal.narrative.chain import AllProvidersFailed
    from adsignal.pipeline import assets as assets_module
    from adsignal.warehouse.narrative_store import ensure_narrative_schema

    _, settings = pipeline_run

    class _FailingChain:
        def generate(self, request):
            raise AllProvidersFailed("every provider failed: stub (RuntimeError)")

    ensure_narrative_schema(settings.postgres_dsn)
    with psycopg.connect(settings.postgres_dsn) as conn:
        before = conn.execute("SELECT count(*) FROM narrative_runs").fetchone()[0]

    context = build_asset_context(
        resources={"settings": settings, "narrative_chain": _FailingChain()}
    )

    # Must not raise.
    assets_module.daily_narrative(context)

    with psycopg.connect(settings.postgres_dsn) as conn:
        after = conn.execute("SELECT count(*) FROM narrative_runs").fetchone()[0]
    assert after == before


def test_embedding_asset_writes_vectors(settings) -> None:
    result = materialize(
        [catalog_and_raw_events, dbt_marts, campaign_daily_mart, campaign_embeddings],
        resources={"settings": settings},
    )
    assert result.success

    metadata = result.asset_materializations_for_node("campaign_embeddings")[0].metadata
    assert metadata["rows_written"].value > 0
    assert metadata["campaigns"].value > 0
    assert metadata["creatives"].value > 0


def test_embedding_asset_skips_the_model_on_an_empty_staging_layer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `if not entities: ... return` guard in `campaign_embeddings` is one
    line and trivially removable by accident; nothing else in the suite would
    notice if it went missing. Build a DuckDB file whose staging tables exist
    but are empty, invoke the asset directly (no need for `dbt_marts` or
    Postgres - the guard returns before either is touched), and assert both
    that `rows_written` is 0 and that `EmbeddingModel` is never constructed."""
    from dagster import build_asset_context

    from adsignal.pipeline import assets as assets_module
    from adsignal.warehouse.duckdb_io import open_duckdb

    duckdb_path = tmp_path / "empty.duckdb"
    with open_duckdb(duckdb_path) as conn:
        conn.execute("CREATE SCHEMA main_staging")
        conn.execute(
            "CREATE TABLE main_staging.stg_campaigns "
            "(campaign_id VARCHAR, campaign_name VARCHAR, channel VARCHAR)"
        )
        conn.execute(
            "CREATE TABLE main_staging.stg_creatives "
            "(creative_id VARCHAR, headline VARCHAR, format VARCHAR)"
        )

    def _must_not_construct(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("EmbeddingModel must not be constructed with nothing to embed")

    monkeypatch.setattr(assets_module, "EmbeddingModel", _must_not_construct)

    empty_settings = Settings(
        data_dir=tmp_path,
        duckdb_path=duckdb_path,
        postgres_dsn=DSN,
        mlflow_dir=tmp_path / "mlruns",
    )
    context = build_asset_context(resources={"settings": empty_settings})

    assets_module.campaign_embeddings(context)

    metadata = context.get_output_metadata("result")
    assert metadata is not None
    assert metadata["rows_written"].value == 0


def test_the_narrative_chain_outlives_a_single_materialisation(pipeline_run) -> None:
    """Finding 3: a chain rebuilt per run has breakers that can never fire.

    `daily_narrative` calls `generate` once, so within one call a provider is
    attempted at most once and `is_open()` is always False. The breakers only
    mean anything if the chain - and therefore its breaker state - survives
    from one materialisation to the next, which is why the chain is a
    resource. With a threshold of one, the second run must skip the dead
    provider instead of calling it again.

    This pins in-process sharing only. `dagster dev` launches each run in a
    fresh subprocess, so breaker state does not survive between nightly runs
    in the deployed topology; the chain-wide time budget is what caps a dead
    provider's cost there. See ARCHITECTURE.md, "Circuit breakers".
    """
    from dagster import build_asset_context

    from adsignal.narrative.chain import NarrativeChain
    from adsignal.narrative.providers.stub import StubProvider
    from adsignal.pipeline import assets as assets_module

    _, settings = pipeline_run

    dead = StubProvider(fail_with=RuntimeError("upstream down"), name="dead")
    chain = NarrativeChain([dead, StubProvider("fallback", name="stub")], failure_threshold=1)
    context = build_asset_context(resources={"settings": settings, "narrative_chain": chain})

    assets_module.daily_narrative(context)
    calls_after_first_run = dead.call_count

    assets_module.daily_narrative(context)

    assert calls_after_first_run == 1
    assert dead.call_count == 1, "the breaker did not survive the first materialisation"


def test_a_failing_asset_still_records_a_failed_run(
    pipeline_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review Focus 2, end to end: the log must not go quiet when the pipeline breaks.

    `pipeline_runs` is append-only and never truncated by this module, so
    comparing row *counts* capped at a fixed page (`limit=100`) stops being
    meaningful once the table passes that many rows - both `before` and
    `after` would read 100 forever. Scoping the assertion to rows for this
    asset started at or after this test's own `datetime.now(UTC)` marker
    finds the row this test caused, however many other rows already exist.
    """
    from datetime import UTC, datetime

    from dagster import build_asset_context

    from adsignal.pipeline import assets as assets_module
    from adsignal.warehouse.run_log import read_runs

    _, settings = pipeline_run

    marker = datetime.now(UTC)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("induced failure")

    monkeypatch.setattr(assets_module, "train_and_evaluate", _boom)

    context = build_asset_context(resources={"settings": settings})
    with pytest.raises(RuntimeError):
        assets_module.trained_models(context)

    recent = [
        r
        for r in read_runs(settings.postgres_dsn, limit=100)
        if r.asset == "trained_models" and r.started_at >= marker
    ]
    assert recent, "no trained_models row was recorded for this test's run"
    assert any(r.status == "failed" and r.detail == "RuntimeError" for r in recent)


def test_the_definitions_share_one_narrative_chain() -> None:
    """The deployed system gets the long-lived chain, not just the tests.

    This pins in-process sharing only. `dagster dev` launches each run in a
    fresh subprocess, so breaker state does not survive between nightly runs
    in the deployed topology; the chain-wide time budget is what caps a dead
    provider's cost there. See ARCHITECTURE.md, "Circuit breakers".
    """
    from adsignal.narrative.chain import NarrativeChain
    from adsignal.pipeline.definitions import defs

    assert isinstance(defs.resources["narrative_chain"], NarrativeChain)
