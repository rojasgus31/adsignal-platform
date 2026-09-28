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
import time
from collections.abc import Callable
from dataclasses import dataclass

import sklearn

from adsignal.config import Settings
from adsignal.ml.registry import loads_classifier, loads_detector
from adsignal.ml.scoring import ScoringModels
from adsignal.warehouse.artifact_store import StoredArtifacts, read_latest_artifacts

_NO_ARTIFACTS_REASON = "no trained model artifacts are available to this API process"

# How long a refusal the deployment can outgrow on its own stays cached before
# the database is consulted again.
#
# Two of the four refusals are transient in that sense: "no artifacts row yet"
# - the state of every deployment between the API starting and the nightly
# training job finishing its first run - and "the database could not be read".
# Caching either for the life of the process would leave an API that started
# before the first training run answering 503 forever, never noticing the
# artifacts the cron writes at 03:00, which is the exact failure this whole
# store exists to prevent.
#
# A minute is long enough that the cost stays far below one round trip per
# request, and short enough that a deployment picks up its first trained run
# within a minute of it being written. Being precise about the bound, because
# an earlier version of this file shipped a comment that promised more than
# the code delivered: it is one round trip per minute per BURST, not per
# minute outright. At each expiry the same unbounded fan-out described above
# applies - every request arriving while that resolve is in flight does its
# own round trip. The other two
# refusals (a scikit-learn version mismatch, a pickle this process cannot
# read) stay cached forever on purpose: only a new image or a new training run
# clears them, and a new image restarts the process anyway.
TRANSIENT_REFUSAL_TTL_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of one attempt to resolve this process's scoring models.

    `models` and `refusal_reason` are mutually exclusive: exactly one is
    `None`, and `__post_init__` enforces that rather than leaving it to this
    sentence. One immutable value rather than two separate attributes on the
    holder, so publishing a resolution - success or failure alike - is a
    single assignment (see `ScoringModelHolder._publish`).

    `expires_at` is the `clock()` reading from which this resolution must be
    derived again, or `None` for one nothing short of a redeploy can change.
    """

    models: ScoringModels | None
    refusal_reason: str | None
    expires_at: float | None = None

    def __post_init__(self) -> None:
        if (self.models is None) == (self.refusal_reason is None):
            raise ValueError("a resolution carries either models or a refusal reason, never both")

    def is_expired(self, now: float) -> bool:
        """Whether this resolution must be derived again rather than served from the cache."""
        return self.expires_at is not None and now >= self.expires_at


class ScoringModelHolder:
    """The process's one `ScoringModels`, resolved from the newest stored run.

    `resolve()` answers with either the models or the reason there are none,
    so the caller can turn a refusal into a 503 naming what is wrong instead
    of a 500 and a traceback. The four refusals are: no artifacts row yet, a
    stored run whose scikit-learn version does not match this process's, a
    pickle this process cannot load for any other reason, and a database this
    process could not read.

    One accessor, not a `get()` plus a separate `reason()`: two calls could
    land either side of a cache expiry and disagree, and a reason-only
    accessor has nothing truthful to say after a successful resolution.
    """

    def __init__(self, settings: Settings, clock: Callable[[], float] = time.monotonic) -> None:
        self._settings = settings
        self._clock = clock
        self._resolution: Resolution | None = None
        self._publish_lock = threading.Lock()

    def resolve(self) -> Resolution:
        """The models this process serves, or the reason it has none.

        Both outcomes are cached, success and failure alike: a deployment
        with no artifacts yet, or one whose serving image has drifted from
        its training image, must not repeat the database round trip on every
        request - that was this holder's own bug. Refusals a deployment can
        outgrow without being redeployed expire after
        `TRANSIENT_REFUSAL_TTL_SECONDS`; the rest are kept for the life of
        the process. See that constant for which are which and why.

        The database read and the unpickling both happen outside
        `self._publish_lock`. Holding a lock across that round trip is
        exactly the mistake `warehouse.pool`'s own docstring warns against,
        and it is what the resolver this replaced did on every request for as
        long as no artifacts existed. The trade is that every caller arriving
        before the first publish resolves for itself: that work is one-time,
        but it is *not* bounded - it is proportional to how many requests
        arrive while the first resolve is in flight, which is whatever the
        server's worker pool allows. Only the first result to reach the lock
        is published, and every call after that returns from the cache with
        no lock and no database call at all.
        """
        now = self._clock()
        cached = self._resolution
        if cached is not None and not cached.is_expired(now):
            return cached
        return self._publish(self._resolve(now), now)

    def _publish(self, resolution: Resolution, now: float) -> Resolution:
        """Install `resolution`, unless a concurrent caller already published a live one.

        The lock covers one attribute read and one assignment - no network
        I/O, no unpickling. Everything a reader can observe is a fully
        constructed frozen value, so a lock-free read in `resolve` sees
        either `None` or a complete resolution, never a half-built one.
        """
        with self._publish_lock:
            published = self._resolution
            if published is None or published.is_expired(now):
                published = resolution
                self._resolution = published
            return published

    def _resolve(self, now: float) -> Resolution:
        """Load both models from the newest stored run, or say why not."""
        try:
            stored = read_latest_artifacts(self._settings.postgres_dsn)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: B036 - re-raised above where it must be
            return self._transient(_unreadable_database_reason(exc), now)

        if stored is None:
            return self._transient(_NO_ARTIFACTS_REASON, now)

        mismatch = _version_mismatch_reason(stored)
        if mismatch is not None:
            return Resolution(models=None, refusal_reason=mismatch)

        detector_blob = _Blob(stored.detector_bytes)
        triage_blob = _Blob(stored.triage_bytes)
        del stored
        return _load_pickled(detector_blob, triage_blob)

    def _transient(self, reason: str, now: float) -> Resolution:
        """A refusal this deployment can outgrow on its own, re-checked after the TTL."""
        return Resolution(
            models=None,
            refusal_reason=reason,
            expires_at=now + TRANSIENT_REFUSAL_TTL_SECONDS,
        )


class _Blob:
    """One pickled blob, handed to its unpickler exactly once and released then.

    A mutable holder rather than a plain `bytes` local because releasing the
    blob is the whole point. `api.model_loader` reads both models out of one
    Postgres row, and a `del` on a callee's own binding cannot drop the last
    reference to a value the caller's frame still holds - that is precisely
    the no-op this class replaces. The caller holds a reference to *this
    object*, not to the bytes, so `take` clearing its own field really is the
    last reference going away, and the detector's blob is freed before the
    classifier's is unpickled rather than both staying resident at once.
    """

    __slots__ = ("_data",)

    def __init__(self, data: bytes) -> None:
        self._data: bytes | None = data

    def take(self) -> bytes:
        """Hand over the bytes and drop this holder's reference to them."""
        data = self._data
        if data is None:
            raise RuntimeError("this blob has already been taken")
        self._data = None
        return data


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


def _load_pickled(detector_blob: _Blob, triage_blob: _Blob) -> Resolution:
    """Unpickle both blobs, turning any failure into a refusal naming only the exception type.

    Never the exception's message or a traceback: either could carry a path,
    a driver-embedded value, or in the worst case bytes from the stored row
    itself, and no secret may reach an API response (see the module's
    binding constraints). The exception type name is enough to distinguish
    "this is a version problem already covered above" from "this pickle is
    simply corrupt" without repeating either.

    Takes each blob from a `_Blob` rather than reading both fields off the
    `StoredArtifacts` value, so that once `loads_detector` returns nothing
    references the detector's raw bytes any more and they are released before
    the classifier's own blob is unpickled. See `_Blob` for why the indirection
    is what makes that true.
    """
    try:
        detector = loads_detector(detector_blob.take())
        classifier = loads_classifier(triage_blob.take())
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: B036 - re-raised above where it must be
        return Resolution(models=None, refusal_reason=_load_failure_reason(exc))
    return Resolution(models=ScoringModels(detector, classifier), refusal_reason=None)


def _load_failure_reason(exc: BaseException) -> str:
    """The refusal text for an unpickling failure: the exception's type name, nothing else."""
    return f"stored model artifacts failed to load ({type(exc).__name__})"


def _unreadable_database_reason(exc: BaseException) -> str:
    """The refusal text for a database this process could not read: the type name, nothing else.

    A connection failure's own message can quote the DSN it failed to connect
    with, and the DSN carries the database password; a type name cannot.
    Refusing here rather than letting the exception escape keeps this holder's
    stated contract - a named 503, never a 500 - true for an unreachable
    database too, which is the one condition the behaviour table this loader
    was written against did not enumerate.
    """
    return f"stored model artifacts could not be read from the database ({type(exc).__name__})"
