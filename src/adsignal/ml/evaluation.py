"""Scoring the models against a held-out future window.

Metrics are computed on a temporal holdout — a window strictly later than
everything the model trained on. Anything else measures the model's ability to
interpolate within data it has seen, which is not the question.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.triage import TriageClassifier

# Chance for 5 balanced classes is 0.20. This gate is deliberately modest: it
# catches a broken pipeline, not a mediocre model.
MINIMUM_MACRO_F1 = 0.45

# A detector with a badly-mismatched contamination assumption (e.g. the old
# hardcoded 0.05 against a mart with a ~29% real anomaly rate) lands around
# 0.20 recall — it misses roughly 4 of every 5 real anomalies while still
# reporting high precision, which reads as "healthy" on a dashboard that only
# shows precision. This floor is set above that failure mode and below what a
# correctly-configured detector clears in practice; see the evaluation report
# for the measured before/after recall this gate is calibrated against.
MINIMUM_DETECTOR_RECALL = 0.5


@dataclass(frozen=True, slots=True)
class TriageMetrics:
    """Classification quality for the cause model on one holdout window."""

    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: dict[str, dict[str, int]]
    n_samples: int

    def meets_gate(self, minimum: float = MINIMUM_MACRO_F1) -> bool:
        """Whether this run clears the CI quality gate."""
        return self.macro_f1 >= minimum


@dataclass(frozen=True, slots=True)
class DetectorMetrics:
    """Detection quality against the generator's ground-truth flags."""

    precision: float
    recall: float
    f1: float
    n_samples: int
    n_flagged: int

    def meets_gate(self, minimum: float = MINIMUM_DETECTOR_RECALL) -> bool:
        """Whether this run clears the CI quality gate."""
        return self.recall >= minimum


def evaluate_triage(classifier: TriageClassifier, holdout: pd.DataFrame) -> TriageMetrics:
    """Score the cause model on the labelled anomalies in `holdout`."""
    labelled = holdout[holdout["is_anomaly"].fillna(False).astype(bool)]
    labelled = labelled[labelled["anomaly_cause"].notna()]
    if labelled.empty:
        raise ValueError("no labelled anomalies in the holdout window; cannot evaluate")

    truth = labelled["anomaly_cause"].to_numpy()
    predicted = classifier.predict(labelled)["cause"].to_numpy()
    labels = sorted(set(truth) | set(predicted))

    per_class = f1_score(truth, predicted, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=labels)

    return TriageMetrics(
        accuracy=float(accuracy_score(truth, predicted)),
        macro_f1=float(f1_score(truth, predicted, average="macro", zero_division=0)),
        per_class_f1={label: float(score) for label, score in zip(labels, per_class, strict=True)},
        confusion={
            actual: {
                predicted_label: int(count)
                for predicted_label, count in zip(labels, row, strict=True)
            }
            for actual, row in zip(labels, matrix, strict=True)
        },
        n_samples=int(len(labelled)),
    )


def evaluate_detector(detector: AnomalyDetector, holdout: pd.DataFrame) -> DetectorMetrics:
    """Score the detector's flags against the generator's ground truth."""
    truth = holdout["is_anomaly"].fillna(False).astype(bool).to_numpy()
    flagged = detector.predict(holdout)
    return DetectorMetrics(
        precision=float(precision_score(truth, flagged, zero_division=0)),
        recall=float(recall_score(truth, flagged, zero_division=0)),
        f1=float(f1_score(truth, flagged, zero_division=0)),
        n_samples=int(len(holdout)),
        n_flagged=int(flagged.sum()),
    )
