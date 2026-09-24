"""Training-run logging and model persistence.

This is the only module that talks to MLflow. Tracking is file-based and local
(`mlruns/`, gitignored) — a tracking server would be infrastructure this
project does not need and cannot host for free.
"""

import os
import pickle
from pathlib import Path

import mlflow

from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.evaluation import DetectorMetrics, TriageMetrics
from adsignal.ml.triage import TriageClassifier

_EXPERIMENT = "adsignal"
_DETECTOR_FILE = "detector.pkl"
_CLASSIFIER_FILE = "triage.pkl"

# MLflow >=3 refuses the plain filesystem tracking backend by default, steering
# users toward a database-backed store. This project deliberately wants local,
# file-based tracking with no server to run — see the module docstring — so
# this is the documented opt-out, not a workaround for a bug.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


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

    detector: AnomalyDetector = pickle.loads(detector_path.read_bytes())  # noqa: S301
    classifier: TriageClassifier = pickle.loads(classifier_path.read_bytes())  # noqa: S301
    return detector, classifier


def log_training_run(
    *,
    tracking_dir: Path,
    detector: AnomalyDetector,
    classifier: TriageClassifier,
    triage_metrics: TriageMetrics,
    detector_metrics: DetectorMetrics,
    params: dict[str, object],
) -> str:
    """Log one training run to MLflow and return its run id."""
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
            }
        )
        for cause, score in triage_metrics.per_class_f1.items():
            mlflow.log_metric(f"triage_f1_{cause}", score)

        artifacts = tracking_dir / "artifacts" / run.info.run_id
        save_models(artifacts, detector, classifier)
        mlflow.log_artifacts(str(artifacts))

        return str(run.info.run_id)
