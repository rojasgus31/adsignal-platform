import numpy as np
import pytest

from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.evaluation import (
    MINIMUM_DETECTOR_AUC,
    MINIMUM_MACRO_F1,
    DetectorMetrics,
    evaluate_detector,
    evaluate_triage,
)
from adsignal.ml.features import build_features
from adsignal.ml.splits import temporal_split
from adsignal.ml.triage import TriageClassifier


@pytest.fixture
def trained(labelled_series):
    features = build_features(labelled_series(240))
    train, holdout = temporal_split(features, holdout_fraction=0.25)
    classifier = TriageClassifier(random_state=0).fit(train)
    return classifier, holdout


def test_triage_metrics_are_in_range(trained) -> None:
    classifier, holdout = trained
    metrics = evaluate_triage(classifier, holdout)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert metrics.n_samples > 0


def test_triage_metrics_report_per_class_scores(trained) -> None:
    classifier, holdout = trained
    metrics = evaluate_triage(classifier, holdout)
    assert metrics.per_class_f1
    for cause, score in metrics.per_class_f1.items():
        assert isinstance(cause, str)
        assert 0.0 <= score <= 1.0


def test_confusion_matrix_totals_match_the_sample_count(trained) -> None:
    classifier, holdout = trained
    metrics = evaluate_triage(classifier, holdout)
    total = sum(sum(row.values()) for row in metrics.confusion.values())
    assert total == metrics.n_samples


def test_evaluation_on_an_unlabelled_holdout_raises(trained) -> None:
    classifier, holdout = trained
    clean = holdout.copy()
    clean["is_anomaly"] = False
    with pytest.raises(ValueError, match="no labelled anomalies"):
        evaluate_triage(classifier, clean)


def test_detector_metrics_are_in_range(labelled_series) -> None:
    features = build_features(labelled_series(240))
    train, holdout = temporal_split(features, holdout_fraction=0.25)
    detector = AnomalyDetector(random_state=0).fit(train)
    metrics = evaluate_detector(detector, holdout)
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert metrics.n_samples == len(holdout)


def test_roc_auc_is_computed_from_the_continuous_score_not_the_boolean_flags(
    labelled_series,
) -> None:
    """A detector whose flags are all wrong but whose ranking is perfect must score well.

    Scoring AUC against `predict()`'s boolean output instead of `score()`'s
    continuous ranking would conflate "the operating point is wrong" with
    "the underlying signal is wrong" - exactly the distinction this gate
    exists to preserve. This fabricates a detector-shaped stand-in whose
    `predict()` deliberately disagrees with a perfectly-ranking `score()`, so
    the two would produce very different AUCs; only the continuous-score
    computation can be correct here.
    """
    features = build_features(labelled_series(240))
    _, holdout = temporal_split(features, holdout_fraction=0.25)
    truth = holdout["is_anomaly"].fillna(False).astype(bool).to_numpy()

    class _StandInDetector:
        def predict(self, frame) -> np.ndarray:  # noqa: ANN001
            # Deliberately the inverse of the truth: every flag is wrong.
            return np.asarray(~truth)

        def score(self, frame) -> np.ndarray:  # noqa: ANN001
            # Perfectly ranks anomalies above normal days.
            return np.asarray(truth, dtype="float64")

    metrics = evaluate_detector(_StandInDetector(), holdout)  # type: ignore[arg-type]
    assert metrics.roc_auc == pytest.approx(1.0)
    # The boolean flags were all wrong, so recall/precision are near zero -
    # confirming AUC could not have been computed from `predict()`'s output.
    assert metrics.recall == pytest.approx(0.0)


def test_split_is_temporal_not_random(labelled_series) -> None:
    """Review Focus 2: evaluation must score a FUTURE window, not a shuffled one."""
    features = build_features(labelled_series(240))
    train, holdout = temporal_split(features, holdout_fraction=0.25)
    assert train["event_date"].max() < holdout["event_date"].min()


def test_minimum_macro_f1_is_a_sane_gate() -> None:
    """The CI gate must be above chance for 5 classes (0.2) and below perfect."""
    assert 0.2 < MINIMUM_MACRO_F1 < 1.0


def test_minimum_detector_auc_is_a_sane_gate() -> None:
    """The gate must reject a no-better-than-random detector (0.5 AUC) but be reachable."""
    assert 0.5 < MINIMUM_DETECTOR_AUC < 1.0


def test_detector_metrics_meets_gate_is_false_below_the_auc_floor() -> None:
    """An AUC of 0.5 is no better than a coin flip; must fail the gate."""
    metrics = DetectorMetrics(
        precision=0.76, recall=0.20, f1=0.32, roc_auc=0.5, n_samples=226, n_flagged=17
    )
    assert metrics.meets_gate() is False


def test_detector_metrics_meets_gate_is_true_above_the_auc_floor() -> None:
    metrics = DetectorMetrics(
        precision=0.8, recall=0.9, f1=0.85, roc_auc=0.9, n_samples=226, n_flagged=70
    )
    assert metrics.meets_gate() is True


def test_detector_metrics_meets_gate_ignores_recall_when_auc_is_high() -> None:
    """Recall alone no longer gates: a low-recall, high-AUC run must still pass.

    This is the case the fix-round rationale is built on - a detector at the
    F1-optimal, base-rate-derived contamination can have modest recall while
    its underlying ranking (AUC) is strong. Gating on AUC must not resurrect
    a hidden recall floor.
    """
    metrics = DetectorMetrics(
        precision=0.35, recall=0.30, f1=0.32, roc_auc=0.76, n_samples=2375, n_flagged=85
    )
    assert metrics.meets_gate() is True


def test_safe_roc_auc_returns_a_neutral_value_on_a_single_class_holdout() -> None:
    """`roc_auc_score` raises on a one-class holdout; the gate must not crash on it.

    A holdout window with zero anomalies (or, degenerately, all anomalies)
    gives no basis to rank anomalous above normal. 0.5 is returned instead of
    propagating the `ValueError` - the neutral value that cannot itself pass
    the gate as a false positive, but does not crash the build either.
    """
    from adsignal.ml.evaluation import _safe_roc_auc

    truth = np.zeros(50, dtype=bool)
    scores = np.random.default_rng(0).normal(size=50)
    assert _safe_roc_auc(truth, scores) == 0.5

    all_true = np.ones(50, dtype=bool)
    assert _safe_roc_auc(all_true, scores) == 0.5
