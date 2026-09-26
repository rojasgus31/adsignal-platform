"""Resolving and holding the serving API's scoring models.

`create_app` builds one of these and puts it on `app.state`, exactly as it
builds one `EmbeddingModel`: the holder is constructed at startup, and the
weights behind it load on first use and are then shared by every request.
Unpickling an IsolationForest of 200 trees plus two gradient-boosted
ensembles per request would be the same mistake as reloading the embedder's
90 MB per `/search`.

Loading lazily rather than eagerly in `create_app` is deliberate on two
counts. The app must start when Postgres is not reachable yet or no model has
been trained - today `create_app` touches no database at all, and `/health`
and every cached read must keep working on a half-built deployment. And the
memory this costs is only paid by a deployment that actually serves `/score`.

Artifacts are read from Postgres (`warehouse.artifact_store`), not from a
filesystem directory: the API and the nightly training job are separate
processes that do not share a disk in production, and a directory path is
only ever valid on the machine that wrote it. See `warehouse.artifact_store`
for the table this reads and why it records scikit-learn's version alongside
the pickled bytes.
"""

import threading
from dataclasses import dataclass

import sklearn

from adsignal.config import Settings
from adsignal.ml.registry import loads_classifier, loads_detector
from adsignal.ml.scoring import ScoringModels
from adsignal.warehouse.artifact_store import StoredArtifacts, read_latest_artifacts

_NO_ARTIFACTS_REASON = "no trained model artifacts are available to this API process"


@dataclass(frozen=True, slots=True)
class _Resolution:
    """The outcome of one attempt to resolve this process's scoring models.

    `models` and `refusal_reason` are mutually exclusive: exactly one is
    `None`. One immutable value rather than two separate attributes on the
    holder, so publishing a resolution - success or failure - is a single
    assignment (see `ScoringModelHolder._resolve_once`).
    """

    models: ScoringModels | None
    refusal_reason: str | None


class ScoringModelHolder:
    """The process's one `ScoringModels`, resolved from the newest stored run.

    `get()` returns `None` when there is nothing to serve: no artifacts row,
    a stored run whose scikit-learn version does not match this process's,
    or a pickle this process cannot load for any other reason.
    `refusal_reason()` names which, so the caller can turn it into a 503
    instead of a 500 and a traceback.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._resolution: _Resolution | None = None
        self._publish_lock = threading.Lock()

    def get(self) -> ScoringModels | None:
        """Return the shared models, resolving them on the first call.

        Both outcomes are cached, success and failure alike: a deployment
        with no artifacts yet, or one whose serving image has drifted from
        its training image, must not repeat the database round trip on
        every request - that was this holder's own bug (see
        `_resolve_once`).
        """
        return self._resolve_once().models

    def refusal_reason(self) -> str:
        """Why `get()` last returned `None`. Meaningless if `get()` returned a value."""
        resolution = self._resolve_once()
        return resolution.refusal_reason or _NO_ARTIFACTS_REASON

    def _resolve_once(self) -> _Resolution:
        """Resolve, or return the cached resolution.

        The database read and the unpickling both happen here, entirely
        outside `self._publish_lock`. Holding a lock across that round trip
        is exactly the mistake `warehouse.pool`'s own docstring warns
        against, and it is what the resolver this replaced did on every
        request for as long as no artifacts existed, because a failed
        resolution was never cached. A race between concurrent first callers
        can resolve more than once - a bounded, one-time cost paid only
        during that race - but only the first result to reach the lock below
        is ever published, and every call after that returns instantly with
        no lock and no database call at all.
        """
        resolution = self._resolution
        if resolution is not None:
            return resolution
        resolution = self._resolve()
        with self._publish_lock:
            if self._resolution is None:
                self._resolution = resolution
            return self._resolution

    def _resolve(self) -> _Resolution:
        """Load both models from the newest stored run, or say why not."""
        stored = read_latest_artifacts(self._settings.postgres_dsn)
        if stored is None:
            return _Resolution(models=None, refusal_reason=_NO_ARTIFACTS_REASON)

        mismatch = _version_mismatch_reason(stored)
        if mismatch is not None:
            return _Resolution(models=None, refusal_reason=mismatch)

        return _load_pickled(stored)


def _version_mismatch_reason(stored: StoredArtifacts) -> str | None:
    """`None` if `stored` was fitted with this process's scikit-learn, else the refusal text.

    Not a 500: on a nightly-cron deployment the serving image and the
    training image will eventually drift, and a pickled estimator is only
    loadable by a compatible scikit-learn. Naming both versions turns that
    drift into a readable refusal instead of a bare unpickling traceback.
    """
    running_version = sklearn.__version__
    if stored.sklearn_version == running_version:
        return None
    return (
        f"stored model artifacts were fitted with scikit-learn {stored.sklearn_version}, "
        f"this process runs {running_version}"
    )


def _load_pickled(stored: StoredArtifacts) -> _Resolution:
    """Unpickle `stored`'s bytes, turning any failure into a refusal naming only the exception type.

    Never the exception's message or a traceback: either could carry a path,
    a driver-embedded value, or in the worst case bytes from the stored row
    itself, and no secret may reach an API response (see the module's
    binding constraints). The exception type name is enough to distinguish
    "this is a version problem already covered above" from "this pickle is
    simply corrupt" without repeating either.

    Unpickles one model at a time rather than both in one call, and drops
    `stored` - the only thing keeping the detector's raw pickled bytes
    reachable, since a frozen dataclass's fields cannot be cleared
    individually - as soon as the detector is built and before the
    classifier's own (larger) blob is unpickled. Measured to matter: holding
    both pickled blobs and both fitted estimators live at once, rather than
    releasing the first blob before allocating for the second, was the
    difference between this loader and the file-based one it replaced
    showing up directly in `scripts/measure_memory.py`'s peak.
    """
    try:
        detector = loads_detector(stored.detector_bytes)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        return _Resolution(models=None, refusal_reason=_load_failure_reason(exc))

    triage_bytes = stored.triage_bytes
    del stored  # drops the last reference to the detector's now-unneeded raw bytes

    try:
        classifier = loads_classifier(triage_bytes)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        return _Resolution(models=None, refusal_reason=_load_failure_reason(exc))
    del triage_bytes

    return _Resolution(models=ScoringModels(detector, classifier), refusal_reason=None)


def _load_failure_reason(exc: Exception) -> str:
    """The refusal text for an unpickling failure: the exception's type name, nothing else."""
    return f"stored model artifacts failed to load ({type(exc).__name__})"
