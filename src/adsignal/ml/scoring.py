"""Scoring one campaign-day with a trained detector/triage pair.

Pure model work: an artifact directory in, a one-row feature frame in, a
verdict out. This module imports nothing from `warehouse`, `pipeline`,
`config` or `api` — it opens no connections, reads no environment and knows
nothing about where its feature row came from. `tests/test_scoring.py` parses
this file and asserts exactly that.

The models take `pandas.DataFrame`s, not mappings: `AnomalyDetector.predict`,
`AnomalyDetector.score` and `TriageClassifier.predict` are all frame-in
methods, fitted on the columns `adsignal.ml.features.FEATURE_COLUMNS`
declares. Scoring therefore takes the same one-row frame the batch path
scores in bulk, so the on-demand answer and the stored nightly answer are
produced by the same code on the same shape of input.

What this module deliberately does NOT check is whether the row's rolling
baselines are real. A campaign-day needs `baseline_window` prior days before
`spend_baseline`, `ctr_ratio` and the rest are anything but NaN, and only the
layer that read the campaign's history knows how many prior days there
actually were. That check lives there, once, where it can name the shortfall
— see `adsignal.warehouse.feature_reads.read_feature_row`.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.registry import load_models
from adsignal.ml.triage import TriageClassifier


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """One campaign-day's verdict.

    `cause`, `confidence`, `severity` and `escalate` are `None` when the
    detector did not flag the row. Triage is fitted only on labelled
    anomalies, so asking it to explain an unremarkable day would return its
    class prior dressed up as a finding.
    """

    anomaly_score: float
    is_anomaly: bool
    cause: str | None = None
    confidence: float | None = None
    severity: int | None = None
    escalate: bool | None = None


class ScoringModels:
    """A loaded detector/triage pair, scoring one feature row at a time.

    Constructed once per process and shared across requests: unpickling both
    estimators per call would reload every tree in the forest on every score.
    """

    def __init__(self, detector: AnomalyDetector, classifier: TriageClassifier) -> None:
        self._detector = detector
        self._classifier = classifier

    @classmethod
    def load(cls, directory: Path) -> "ScoringModels":
        """Load both estimators from a registry run's artifact directory.

        Raises `FileNotFoundError` if either artifact is missing, which is
        `registry.load_models`' contract — a half-present run is a broken run,
        not a reason to score with one model.
        """
        detector, classifier = load_models(directory)
        return cls(detector, classifier)

    def score(self, features: pd.DataFrame) -> ScoreResult:
        """Return the verdict for the single row in `features`.

        Raises `ValueError` if `features` does not hold exactly one row, or if
        it is missing a column either model was fitted on. Both estimators
        raise on missing columns themselves rather than reordering or
        imputing, so a mismatch surfaces as a refusal, never as a confident
        number computed from the wrong columns.
        """
        _require_single_row(features)

        anomaly_score = float(self._detector.score(features)[0])
        is_anomaly = bool(self._detector.predict(features)[0])
        if not is_anomaly:
            return ScoreResult(anomaly_score=anomaly_score, is_anomaly=False)

        triage = self._classifier.predict(features).iloc[0]
        return ScoreResult(
            anomaly_score=anomaly_score,
            is_anomaly=True,
            cause=str(triage["cause"]),
            confidence=float(triage["confidence"]),
            severity=int(triage["severity"]),
            escalate=bool(triage["escalate"]),
        )


def _require_single_row(features: pd.DataFrame) -> None:
    """A verdict describes one campaign-day, so exactly one row may be given."""
    if len(features) != 1:
        raise ValueError(f"score() takes exactly one feature row, got {len(features)}")
