"""`model_artifacts` - the table `POST /score` reads models from in production.

The API and the nightly training job do not share a filesystem, so a
directory path is not something the API can act on; pickled bytes in a table
both processes already connect to are. These tests cover the store itself
(round trip, an empty table, replacing a run) and the loader behaviour it
exists to make possible: a version mismatch or a corrupt pickle must become a
named refusal, never a raised exception or a leaked DSN.
"""

import os

import psycopg
import pytest
import sklearn

from adsignal.api.model_loader import ScoringModelHolder
from adsignal.config import Settings
from adsignal.warehouse.artifact_store import (
    StoredArtifacts,
    ensure_artifact_schema,
    read_latest_artifacts,
    write_artifacts,
)

DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)

_DETECTOR_BYTES = b"pretend-detector-pickle"
_TRIAGE_BYTES = b"pretend-triage-pickle"


@pytest.fixture(autouse=True)
def clean():
    ensure_artifact_schema(DSN)
    with psycopg.connect(DSN) as conn:
        conn.execute("TRUNCATE TABLE model_artifacts")
        conn.commit()
    yield


def _write(run_id: str = "run_1", *, sklearn_version: str = "1.5.2") -> None:
    write_artifacts(
        DSN,
        run_id,
        _DETECTOR_BYTES,
        _TRIAGE_BYTES,
        sklearn_version=sklearn_version,
        python_version="3.12.0",
    )


def test_round_trip() -> None:
    _write("run_1", sklearn_version="1.5.2")
    stored = read_latest_artifacts(DSN)

    assert stored is not None
    assert stored.run_id == "run_1"
    assert stored.detector_bytes == _DETECTOR_BYTES
    assert stored.triage_bytes == _TRIAGE_BYTES
    assert stored.sklearn_version == "1.5.2"
    assert stored.python_version == "3.12.0"


def test_reading_an_empty_table_returns_none() -> None:
    assert read_latest_artifacts(DSN) is None


def test_a_second_write_for_the_same_run_id_replaces_it_rather_than_duplicating() -> None:
    _write("run_1", sklearn_version="1.5.2")
    write_artifacts(
        DSN,
        "run_1",
        b"newer-detector",
        b"newer-triage",
        sklearn_version="1.6.0",
        python_version="3.12.1",
    )

    with psycopg.connect(DSN) as conn:
        count = conn.execute("SELECT count(*) FROM model_artifacts").fetchone()[0]
    assert count == 1

    stored = read_latest_artifacts(DSN)
    assert stored is not None
    assert stored.detector_bytes == b"newer-detector"
    assert stored.sklearn_version == "1.6.0"


def test_the_newest_written_run_wins_when_several_exist() -> None:
    _write("run_1")
    write_artifacts(
        DSN,
        "run_2",
        b"second-detector",
        b"second-triage",
        sklearn_version="1.5.2",
        python_version="3.12.0",
    )

    stored = read_latest_artifacts(DSN)
    assert stored is not None and stored.run_id == "run_2"


def test_schema_creation_is_idempotent() -> None:
    ensure_artifact_schema(DSN)
    ensure_artifact_schema(DSN)


def test_a_version_mismatch_is_refused_with_the_reason_not_an_exception() -> None:
    """Absorbs the deferred finding: a drifted scikit-learn must not become a 500."""
    _write("run_1", sklearn_version="0.0.1-does-not-exist")
    holder = ScoringModelHolder(Settings(postgres_dsn=DSN))

    assert holder.get() is None
    reason = holder.refusal_reason()
    assert "0.0.1-does-not-exist" in reason
    assert sklearn.__version__ in reason


def test_a_pickle_this_process_cannot_load_is_refused_by_exception_type_only() -> None:
    """Garbage bytes fail unpickling for a reason that is not a version mismatch."""
    _write("run_1", sklearn_version=sklearn.__version__)
    holder = ScoringModelHolder(Settings(postgres_dsn=DSN))

    assert holder.get() is None
    reason = holder.refusal_reason()
    assert "failed to load" in reason
    # The exception's type name is allowed; its message and any traceback are not.
    assert "pretend-detector-pickle" not in reason


def test_no_artifacts_at_all_is_refused_by_name() -> None:
    holder = ScoringModelHolder(Settings(postgres_dsn=DSN))
    assert holder.get() is None
    assert "no trained model artifacts" in holder.refusal_reason()


@pytest.mark.parametrize("sklearn_version", ["0.0.1-does-not-exist", "1.5.2"])
def test_the_dsn_never_appears_in_a_refusal_message(sklearn_version: str) -> None:
    """No secret may reach an API response - the DSN carries the database password."""
    _write("run_1", sklearn_version=sklearn_version)
    holder = ScoringModelHolder(Settings(postgres_dsn=DSN))

    holder.get()
    reason = holder.refusal_reason()

    assert DSN not in reason
    assert "adsignal:adsignal" not in reason


def test_a_resolution_is_cached_so_a_second_call_does_not_hit_the_database() -> None:
    """Both outcomes - success and refusal - must be cached (the second deferred finding)."""
    holder = ScoringModelHolder(Settings(postgres_dsn=DSN))
    assert holder.get() is None
    first_reason = holder.refusal_reason()

    # Writing a real row after the first resolution must not change the answer:
    # a cached negative result is not re-checked against the database.
    _write("run_1")
    assert holder.get() is None
    assert holder.refusal_reason() == first_reason


def test_stored_artifacts_is_a_frozen_value() -> None:
    stored = StoredArtifacts(
        run_id="run_1",
        detector_bytes=b"d",
        triage_bytes=b"t",
        sklearn_version="1.5.2",
        python_version="3.12.0",
    )
    with pytest.raises((AttributeError, TypeError)):
        stored.run_id = "run_2"  # type: ignore[misc]
