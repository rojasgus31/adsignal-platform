"""Resolving and holding the serving API's scoring models.

`create_app` builds one of these and puts it on `app.state`, exactly as it
builds one `EmbeddingModel`: the holder is constructed at startup, and the
weights behind it load on first use and are then shared by every request.
Unpickling an IsolationForest of 200 trees plus two gradient-boosted
ensembles per request would be the same mistake as reloading the embedder's
90 MB per `/search`.

Loading lazily rather than eagerly in `create_app` is deliberate on two
counts. The app must start when Postgres is not reachable yet or no model has
been trained — today `create_app` touches no database at all, and `/health`
and every cached read must keep working on a half-built deployment. And the
memory this costs is only paid by a deployment that actually serves `/score`.
"""

import threading

from adsignal.config import Settings
from adsignal.ml.scoring import ScoringModels
from adsignal.warehouse.serving_reads import read_latest_artifact_dir


class ScoringModelHolder:
    """The process's one `ScoringModels`, resolved from the newest training run.

    `get()` returns None when there is nothing to serve: no training run
    recorded, or a recorded run whose artifacts are not on this filesystem.
    The caller turns that into a refusal rather than a guess.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loaded: ScoringModels | None = None
        self._load_lock = threading.Lock()

    def get(self) -> ScoringModels | None:
        """Return the shared models, loading them on the first successful call.

        Serialised, for the reason `EmbeddingModel` serialises its own load:
        several concurrent first requests would otherwise each unpickle their
        own copy of both estimators inside one small container.

        A failed resolution is not cached. A deployment whose pipeline has not
        finished its first training run starts serving `/score` as soon as it
        has, without a restart; the endpoint's rate limit bounds what that
        retry costs.
        """
        if self._loaded is not None:
            return self._loaded
        with self._load_lock:
            if self._loaded is None:
                self._loaded = self._resolve()
        return self._loaded

    def _resolve(self) -> ScoringModels | None:
        """Load both models from the newest run's artifact directory, if there is one."""
        directory = read_latest_artifact_dir(self._settings.postgres_dsn)
        if directory is None:
            return None
        try:
            return ScoringModels.load(directory)
        except FileNotFoundError:
            # The directory exists but is missing an artifact: a half-written
            # or partially cleaned run. Serving with one of the two models is
            # not an option, and neither is a 500 on every subsequent call.
            return None
