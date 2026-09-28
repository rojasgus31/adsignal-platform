"""`model_artifacts` - the table `POST /score` reads models from in production.

The API and the nightly training job do not share a filesystem, so a
directory path is not something the API can act on; pickled bytes in a table
both processes already connect to are. These tests cover the store itself
(round trip, an empty table, replacing a run, concurrent DDL) and the loader
behaviour it exists to make possible: a version mismatch, an unreadable
database or a corrupt pickle must become a named refusal, never a raised
exception or a leaked DSN - and the refusals a deployment can outgrow on its
own must not outlive the condition that caused them.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
import sklearn

from adsignal.api import model_loader
from adsignal.api.model_loader import (
    TRANSIENT_REFUSAL_TTL_SECONDS,
    Resolution,
    ScoringModelHolder,
)
from adsignal.config import Settings
from adsignal.warehouse.artifact_store import (
    StoredArtifacts,
    ensure_artifact_schema,
    read_latest_artifacts,
    write_artifacts,
)
from adsignal.warehouse.schema import reset_schema_memo

DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)

# Port 1 is reserved and nothing listens on it, so connecting is refused
# immediately rather than hanging: an unreachable database, cheaply.
UNREACHABLE_DSN = "postgresql://adsignal:adsignal@127.0.0.1:1/adsignal_test"

_DETECTOR_BYTES = b"pretend-detector-pickle"
_TRIAGE_BYTES = b"pretend-triage-pickle"

_CONCURRENT_CALLERS = 16


class _FakeClock:
    """A hand-wound `clock()` for `ScoringModelHolder`, so TTLs are tested without sleeping."""

    def __init__(self) -> None:
        self._now = 1_000.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture(autouse=True)
def clean():
    ensure_artifact_schema(DSN)
    _truncate()
    yield


def _truncate() -> None:
    with psycopg.connect(DSN) as conn:
        conn.execute("TRUNCATE TABLE model_artifacts")
        conn.commit()


def _write(run_id: str = "run_1", *, sklearn_version: str = "1.5.2") -> None:
    write_artifacts(
        DSN,
        run_id,
        _DETECTOR_BYTES,
        _TRIAGE_BYTES,
        sklearn_version=sklearn_version,
        python_version="3.12.0",
    )


def _holder(dsn: str = DSN, clock: _FakeClock | None = None) -> ScoringModelHolder:
    settings = Settings(postgres_dsn=dsn)
    return ScoringModelHolder(settings) if clock is None else ScoringModelHolder(settings, clock)


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


def test_concurrent_schema_creation_against_a_tableless_database_all_succeed() -> None:
    """The DDL race an earlier task shipped: 16 first-ever callers, no table, no `UniqueViolation`.

    Postgres does not serialise concurrent `CREATE TABLE IF NOT EXISTS`; the
    losers abort on a `pg_catalog` index. `apply_schema`'s advisory lock is
    what makes this safe, and the memo has to be reset or the table's
    creation is never actually attempted more than once.
    """
    with psycopg.connect(DSN) as conn:
        conn.execute("DROP TABLE IF EXISTS model_artifacts")
        conn.commit()
    reset_schema_memo(DSN)

    with ThreadPoolExecutor(max_workers=_CONCURRENT_CALLERS) as pool:
        futures = [pool.submit(ensure_artifact_schema, DSN) for _ in range(_CONCURRENT_CALLERS)]
    failures = [future.exception() for future in futures if future.exception() is not None]

    assert failures == []
    with psycopg.connect(DSN) as conn:
        assert conn.execute("SELECT to_regclass('model_artifacts')").fetchone()[0] is not None


def test_concurrent_first_callers_all_receive_the_one_published_resolution() -> None:
    """The resolve-outside/publish-under-lock shape: one winner, no torn state, no exception."""
    holder = _holder()

    with ThreadPoolExecutor(max_workers=_CONCURRENT_CALLERS) as pool:
        resolutions = list(pool.map(lambda _: holder.resolve(), range(_CONCURRENT_CALLERS)))

    assert len({id(resolution) for resolution in resolutions}) == 1
    assert all(resolution.models is None for resolution in resolutions)
    assert all(resolution.refusal_reason is not None for resolution in resolutions)


def test_a_version_mismatch_is_refused_with_the_reason_not_an_exception() -> None:
    """Absorbs the deferred finding: a drifted scikit-learn must not become a 500."""
    _write("run_1", sklearn_version="0.0.1-does-not-exist")

    resolution = _holder().resolve()

    assert resolution.models is None
    reason = resolution.refusal_reason
    assert reason is not None
    assert "0.0.1-does-not-exist" in reason
    assert sklearn.__version__ in reason


def test_a_pickle_this_process_cannot_load_is_refused_by_exception_type_only() -> None:
    """Garbage bytes fail unpickling for a reason that is not a version mismatch."""
    _write("run_1", sklearn_version=sklearn.__version__)

    resolution = _holder().resolve()

    assert resolution.models is None
    reason = resolution.refusal_reason
    assert reason is not None and "failed to load" in reason
    # The exception's type name is allowed; its message and any traceback are not.
    assert "pretend-detector-pickle" not in reason


def test_no_artifacts_at_all_is_refused_by_name() -> None:
    resolution = _holder().resolve()

    assert resolution.models is None
    assert resolution.refusal_reason is not None
    assert "no trained model artifacts" in resolution.refusal_reason


def test_an_unreadable_database_is_refused_by_exception_type_rather_than_raised() -> None:
    """A Postgres outage must be a named 503 like every other refusal, not a 500."""
    resolution = _holder(UNREACHABLE_DSN).resolve()

    assert resolution.models is None
    reason = resolution.refusal_reason
    assert reason is not None and "could not be read from the database" in reason
    assert UNREACHABLE_DSN not in reason
    assert "adsignal:adsignal" not in reason


@pytest.mark.parametrize("sklearn_version", ["0.0.1-does-not-exist", "1.5.2"])
def test_the_dsn_never_appears_in_a_refusal_message(sklearn_version: str) -> None:
    """No secret may reach an API response - the DSN carries the database password."""
    _write("run_1", sklearn_version=sklearn_version)

    reason = _holder().resolve().refusal_reason

    assert reason is not None
    assert DSN not in reason
    assert "adsignal:adsignal" not in reason


def test_a_resolution_is_cached_so_a_second_call_does_not_hit_the_database() -> None:
    """Both outcomes - success and refusal - are cached (the second deferred finding)."""
    clock = _FakeClock()
    holder = _holder(clock=clock)
    first = holder.resolve()
    assert first.models is None

    # Writing a real row while the refusal is still live must not change the
    # answer: within its TTL a cached refusal is not re-checked at all.
    _write("run_1")
    clock.advance(TRANSIENT_REFUSAL_TTL_SECONDS / 2)

    assert holder.resolve() is first


def test_the_no_artifacts_refusal_is_re_checked_once_its_ttl_has_passed() -> None:
    """An API started before the first training run must notice the artifacts the cron writes."""
    clock = _FakeClock()
    holder = _holder(clock=clock)
    first_reason = holder.resolve().refusal_reason
    assert first_reason is not None and "no trained model artifacts" in first_reason

    _write("run_1", sklearn_version=sklearn.__version__)
    clock.advance(TRANSIENT_REFUSAL_TTL_SECONDS + 1)

    # The row is garbage bytes, so the new answer is a *different* refusal -
    # which is the proof that the database was actually read again.
    second_reason = holder.resolve().refusal_reason
    assert second_reason is not None and "failed to load" in second_reason


def test_a_version_mismatch_refusal_is_never_re_checked() -> None:
    """Only a new image or a new training run can clear it, and either restarts the process."""
    clock = _FakeClock()
    _write("run_1", sklearn_version="0.0.1-does-not-exist")
    holder = _holder(clock=clock)
    first = holder.resolve()

    _truncate()
    clock.advance(TRANSIENT_REFUSAL_TTL_SECONDS * 100)

    assert holder.resolve() is first


def test_the_detector_blob_is_released_before_the_classifier_is_unpickled(monkeypatch) -> None:
    """The mechanism `_Blob` exists for, asserted rather than claimed in a docstring.

    A `del` of the callee's own binding could not do this: the caller's frame
    still held the `StoredArtifacts` value, so both blobs stayed reachable for
    the whole call.
    """
    released: list[str] = []
    released_when_classifier_ran: list[list[str]] = []

    class _TrackedBytes(bytes):
        def __del__(self) -> None:
            released.append("detector blob")

    def fake_loads_detector(data: bytes) -> str:
        return "detector"

    def fake_loads_classifier(data: bytes) -> str:
        released_when_classifier_ran.append(list(released))
        return "classifier"

    monkeypatch.setattr(model_loader, "loads_detector", fake_loads_detector)
    monkeypatch.setattr(model_loader, "loads_classifier", fake_loads_classifier)

    resolution = model_loader._load_pickled(
        model_loader._Blob(_TrackedBytes(b"detector")),
        model_loader._Blob(b"triage"),
    )

    assert resolution.models is not None
    assert released_when_classifier_ran == [["detector blob"]]


def test_a_blob_cannot_be_taken_twice() -> None:
    blob = model_loader._Blob(b"detector")
    assert blob.take() == b"detector"
    with pytest.raises(RuntimeError):
        blob.take()


def test_a_resolution_carries_either_models_or_a_reason_but_never_both_or_neither() -> None:
    with pytest.raises(ValueError):
        Resolution(models=None, refusal_reason=None)
    with pytest.raises(ValueError):
        Resolution(models=object(), refusal_reason="both")  # type: ignore[arg-type]


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
