"""Scoring the models against a held-out future window.

Metrics are computed on a temporal holdout — a window strictly later than
everything the model trained on. Anything else measures the model's ability to
interpolate within data it has seen, which is not the question.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.triage import TriageClassifier

# Chance for 5 balanced classes is 0.20. This gate is deliberately modest: it
# catches a broken pipeline, not a mediocre model.
MINIMUM_MACRO_F1 = 0.45

# ROC AUC, not recall, is the CI gate for the detector.
#
# Recall is threshold- and base-rate-dependent: the same detector measured
# 0.60 recall on a mart with a ~29% anomaly rate and 0.30 on one with ~5%,
# because IsolationForest's contamination (and therefore its flagging
# threshold) is derived from the training base rate — see
# `AnomalyDetector._resolve_contamination`. A fixed recall floor is not a
# regression gate under that design; it is a number that moves whenever the
# dataset's anomaly rate moves, with no change to the model at all. A
# contamination sweep on real pipeline data confirmed this isn't a tuning
# problem either: F1 is maximised at the derived (base-rate) contamination,
# and reaching recall 0.5 there costs more than half the detector's
# precision.
#
# AUC asks a different, threshold-independent question: does the detector's
# continuous anomaly score still rank real anomalies above normal days at
# all? That's what a regression gate should ask - not "does today's operating
# point clear an arbitrary bar", but "has the underlying signal degraded".
# 0.65 sits comfortably below the ~0.76 AUC measured on real pipeline data
# and comfortably above 0.5 (no better than random).
MINIMUM_DETECTOR_AUC = 0.65


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
    """Detection quality against the generator's ground-truth flags.

    `precision`, `recall` and `f1` are still reported - they remain useful for
    understanding today's operating point - but `roc_auc` is what
    `meets_gate` checks, because it alone is independent of the dataset's
    anomaly rate. See `MINIMUM_DETECTOR_AUC` for why.
    """

    precision: float
    recall: float
    f1: float
    roc_auc: float
    n_samples: int
    n_flagged: int

    def meets_gate(self, minimum: float = MINIMUM_DETECTOR_AUC) -> bool:
        """Whether this run clears the CI quality gate."""
        return self.roc_auc >= minimum


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
    """Score the detector's flags against the generator's ground truth.

    `roc_auc` is computed from the detector's continuous `score()` output, not
    its boolean `predict()` flags - AUC ranks every possible threshold, so
    scoring it against a single already-thresholded prediction would silently
    collapse it to something else entirely.
    """
    truth = holdout["is_anomaly"].fillna(False).astype(bool).to_numpy()
    flagged = detector.predict(holdout)
    scores = detector.score(holdout)
    return DetectorMetrics(
        precision=float(precision_score(truth, flagged, zero_division=0)),
        recall=float(recall_score(truth, flagged, zero_division=0)),
        f1=float(f1_score(truth, flagged, zero_division=0)),
        roc_auc=_safe_roc_auc(truth, scores),
        n_samples=int(len(holdout)),
        n_flagged=int(flagged.sum()),
    )


def _safe_roc_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    """Return the ROC AUC of `scores` against `truth`, or 0.5 if undefined.

    A holdout with only one class present in `truth` - no anomalies at all,
    or degenerately, all anomalies - gives no basis to rank anomalous above
    normal, so there is no meaningful AUC to report. Depending on the sklearn
    version, `roc_auc_score` either raises `ValueError` or returns `nan` (with
    a warning) for this case; both are treated the same way here. 0.5 (no
    better than a coin flip) is the neutral value: it will not itself pass
    the gate as a false positive, but it also cannot report a failing score
    the data never demonstrated.
    """
    if len(np.unique(truth)) < 2:
        return 0.5
    try:
        auc = float(roc_auc_score(truth, scores))
    except ValueError:
        return 0.5
    return 0.5 if np.isnan(auc) else auc
