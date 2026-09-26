"""Training-run logging and model persistence.

This is the only module that talks to MLflow. Tracking is file-based and local
(`mlruns/`, gitignored) — a tracking server would be infrastructure this
project does not need and cannot host for free.
"""

import os
import pickle
from pathlib import Path

from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.evaluation import DetectorMetrics, TriageMetrics
from adsignal.ml.triage import TriageClassifier

_EXPERIMENT = "adsignal"
_DETECTOR_FILE = "detector.pkl"
_CLASSIFIER_FILE = "triage.pkl"


def save_models(directory: Path, detector: AnomalyDetector, classifier: TriageClassifier) -> None:
    """Persist both fitted models to `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _DETECTOR_FILE).write_bytes(pickle.dumps(detector))
    (directory / _CLASSIFIER_FILE).write_bytes(pickle.dumps(classifier))


def load_models(directory: Path) -> tuple[AnomalyDetector, TriageClassifier]:
    """Load both models from `directory`. Raises if either is missing."""
    detector_path = directory / _DETECTOR_FILE
    classifier_path = directory / _CLASSIFIER_FILE
    for path in (detector_path, classifier_path):
        if not path.exists():
            raise FileNotFoundError(f"no model artifact at {path}")

    # Bandit's pickle-deserialization rule isn't in this project's ruff
    # `select`, so silencing it here would suppress nothing that is actually
    # checked - these artifacts are only ever the models this same process
    # just wrote to `directory` in `save_models`, never untrusted input.
    detector: AnomalyDetector = pickle.loads(detector_path.read_bytes())
    classifier: TriageClassifier = pickle.loads(classifier_path.read_bytes())
    return detector, classifier


def dumps_models(detector: AnomalyDetector, classifier: TriageClassifier) -> tuple[bytes, bytes]:
    """Serialize both fitted models to bytes, in the same format `save_models` writes to disk.

    The one place this project pickles a model for storage other than
    `save_models` itself, so the two never drift into incompatible formats.
    `warehouse.artifact_store` calls this and stores the bytes it gets back
    without importing `pickle` itself - pickling stays owned here, not
    scattered across every place a model gets persisted.
    """
    return pickle.dumps(detector), pickle.dumps(classifier)


def loads_detector(data: bytes) -> AnomalyDetector:
    """Deserialize one detector from bytes.

    Split from the classifier's own `loads_classifier` (rather than one
    combined call taking both blobs) so a caller holding two large pickled
    blobs - `api.model_loader`, reading both out of one Postgres row - can
    drop the first blob's last reference before the second is ever
    unpickled, instead of both staying resident for the whole call. Raises
    whatever `pickle.loads` raises for data it cannot read - most often an
    incompatible scikit-learn version; the caller decides what that means
    for serving.
    """
    detector: AnomalyDetector = pickle.loads(data)
    return detector


def loads_classifier(data: bytes) -> TriageClassifier:
    """Deserialize one classifier from bytes. See `loads_detector`."""
    classifier: TriageClassifier = pickle.loads(data)
    return classifier


def loads_models(
    detector_bytes: bytes, classifier_bytes: bytes
) -> tuple[AnomalyDetector, TriageClassifier]:
    """Deserialize both models from bytes, the counterpart to `dumps_models`.

    A convenience wrapper over `loads_detector`/`loads_classifier` for a
    caller that already holds both blobs and has no reason to drop either
    early (this module's own round-trip test, and anywhere else two small
    blobs are already both in memory anyway). `api.model_loader` calls the
    two single-model functions directly instead, precisely because it does
    have a reason to drop the first blob early - see `loads_detector`.
    """
    detector = loads_detector(detector_bytes)
    classifier = loads_classifier(classifier_bytes)
    return detector, classifier


def log_training_run(
    *,
    tracking_dir: Path,
    detector: AnomalyDetector,
    classifier: TriageClassifier,
    triage_metrics: TriageMetrics,
    detector_metrics: DetectorMetrics,
    params: dict[str, object],
) -> tuple[str, Path]:
    """Log one training run to MLflow; return its run id and artifact directory.

    The artifact directory is returned (not just the run id) so a caller can
    persist it alongside the run in `model_metrics` - the run registry
    (Postgres, shared) and the artifact store (`tracking_dir`, per-`Settings`)
    are otherwise two independent sources of truth that can silently
    disagree about where a run's models live. See
    `adsignal.pipeline.assets._latest_run`.
    """
    # MLflow >=3 refuses the plain filesystem tracking backend by default,
    # steering users toward a database-backed store. This project
    # deliberately wants local, file-based tracking with no server to run —
    # see the module docstring — so this is the documented opt-out, not a
    # workaround for a bug. Set here, immediately before the call that needs
    # it, rather than at import time: importing this module (transitively,
    # just importing `adsignal.pipeline.definitions`) must not mutate the
    # process environment for every embedding host.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    # Imported here, not at module scope, for the same reason the environment
    # variable above is set here: importing this module must stay cheap.
    # `save_models`/`load_models` are pure `pickle` and need nothing from
    # MLflow, and the API process imports them (via `adsignal.ml.scoring`) to
    # serve `POST /score`. A module-scope `import mlflow` costs that process a
    # measured ~136 MB of resident memory it never uses, inside a 512 MB
    # container - see `scripts/measure_memory.py`. Only this function, which
    # runs in the pipeline, actually needs MLflow.
    import mlflow

    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(_EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(
            {
                "triage_accuracy": triage_metrics.accuracy,
                "triage_macro_f1": triage_metrics.macro_f1,
                "triage_n_samples": float(triage_metrics.n_samples),
                "detector_precision": detector_metrics.precision,
                "detector_recall": detector_metrics.recall,
                "detector_f1": detector_metrics.f1,
                "detector_roc_auc": detector_metrics.roc_auc,
            }
        )
        for cause, score in triage_metrics.per_class_f1.items():
            mlflow.log_metric(f"triage_f1_{cause}", score)

        artifacts = tracking_dir / "artifacts" / run.info.run_id
        save_models(artifacts, detector, classifier)
        mlflow.log_artifacts(str(artifacts))

        return str(run.info.run_id), artifacts
