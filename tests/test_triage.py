import numpy as np
import pandas as pd
import pytest

from adsignal.generator.models import ANOMALY_CAUSES
from adsignal.ml.features import build_features
from adsignal.ml.triage import ESCALATE_SEVERITY, TRIAGE_FEATURES, TriageClassifier


def test_classifier_learns_the_causes(labelled_series) -> None:
    features = build_features(labelled_series())
    classifier = TriageClassifier(random_state=0).fit(features)
    anomalies = features[features["is_anomaly"]]
    predictions = classifier.predict(anomalies)
    accuracy = (predictions["cause"].to_numpy() == anomalies["anomaly_cause"].to_numpy()).mean()
    assert accuracy > 0.5, f"accuracy {accuracy} is no better than guessing among 5 classes"


def test_predict_returns_the_documented_columns(labelled_series) -> None:
    features = build_features(labelled_series())
    classifier = TriageClassifier(random_state=0).fit(features)
    predictions = classifier.predict(features[features["is_anomaly"]])
    assert list(predictions.columns) == ["cause", "confidence", "severity", "escalate"]


def test_every_predicted_cause_is_a_known_cause(labelled_series) -> None:
    features = build_features(labelled_series())
    classifier = TriageClassifier(random_state=0).fit(features)
    predictions = classifier.predict(features[features["is_anomaly"]])
    assert set(predictions["cause"]) <= set(ANOMALY_CAUSES)


def test_confidence_is_a_probability(labelled_series) -> None:
    features = build_features(labelled_series())
    classifier = TriageClassifier(random_state=0).fit(features)
    predictions = classifier.predict(features[features["is_anomaly"]])
    assert predictions["confidence"].between(0.0, 1.0).all()


def test_escalate_matches_the_severity_threshold(labelled_series) -> None:
    features = build_features(labelled_series())
    classifier = TriageClassifier(random_state=0).fit(features)
    predictions = classifier.predict(features[features["is_anomaly"]])
    expected = predictions["severity"] >= ESCALATE_SEVERITY
    pd.testing.assert_series_equal(predictions["escalate"], expected, check_names=False)


def test_fit_rejects_single_class_training_set(flat_series) -> None:
    """Review Focus 5: one cause in the window must fail loudly.

    Fitting anyway produces a model that always predicts that class and reports
    100% accuracy, which is worse than an error because nobody notices.
    """
    frame = flat_series(60)
    for index in range(5, 60, 6):
        frame.loc[index, "anomaly_cause"] = "budget_pacing"
        frame.loc[index, "anomaly_severity"] = 3
        frame.loc[index, "is_anomaly"] = True
    features = build_features(frame)
    with pytest.raises(ValueError, match="at least 2 distinct causes"):
        TriageClassifier().fit(features)


def test_fit_rejects_a_training_set_with_no_anomalies(flat_series) -> None:
    features = build_features(flat_series(60))
    with pytest.raises(ValueError, match="no labelled anomalies"):
        TriageClassifier().fit(features)


def test_predict_before_fit_raises(labelled_series) -> None:
    features = build_features(labelled_series())
    with pytest.raises(RuntimeError, match="not fitted"):
        TriageClassifier().predict(features)


def test_nan_features_are_handled_natively(labelled_series) -> None:
    """HistGradientBoosting handles NaN; no imputation should be needed."""
    features = build_features(labelled_series())
    classifier = TriageClassifier(random_state=0).fit(features)
    anomalies = features[features["is_anomaly"]].copy()
    anomalies.loc[anomalies.index[0], "spend_ratio"] = np.nan
    predictions = classifier.predict(anomalies)
    assert len(predictions) == len(anomalies)
    assert predictions["cause"].notna().all()


def test_triage_features_are_all_produced_by_build_features(labelled_series) -> None:
    features = build_features(labelled_series())
    for column in TRIAGE_FEATURES:
        assert column in features.columns
