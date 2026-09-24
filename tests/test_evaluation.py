import pytest

from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.evaluation import (
    MINIMUM_DETECTOR_RECALL,
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
    assert metrics.n_samples == len(holdout)


def test_split_is_temporal_not_random(labelled_series) -> None:
    """Review Focus 2: evaluation must score a FUTURE window, not a shuffled one."""
    features = build_features(labelled_series(240))
    train, holdout = temporal_split(features, holdout_fraction=0.25)
    assert train["event_date"].max() < holdout["event_date"].min()


def test_minimum_macro_f1_is_a_sane_gate() -> None:
    """The CI gate must be above chance for 5 classes (0.2) and below perfect."""
    assert 0.2 < MINIMUM_MACRO_F1 < 1.0


def test_minimum_detector_recall_is_a_sane_gate() -> None:
    """The gate must reject the old-default failure mode (~0.20 recall) but be reachable."""
    assert 0.2 < MINIMUM_DETECTOR_RECALL < 1.0


def test_detector_metrics_meets_gate_is_false_at_the_old_failure_mode_recall() -> None:
    """Recall 0.20 is exactly what the mismatched-contamination bug produced; must fail the gate."""
    metrics = DetectorMetrics(precision=0.76, recall=0.20, f1=0.32, n_samples=226, n_flagged=17)
    assert metrics.meets_gate() is False


def test_detector_metrics_meets_gate_is_true_above_the_floor() -> None:
    metrics = DetectorMetrics(precision=0.8, recall=0.9, f1=0.85, n_samples=226, n_flagged=70)
    assert metrics.meets_gate() is True
