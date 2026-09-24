import pandas as pd

from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.evaluation import evaluate_detector, evaluate_triage
from adsignal.ml.features import build_features
from adsignal.ml.registry import load_models, log_training_run, save_models
from adsignal.ml.splits import temporal_split
from adsignal.ml.triage import TriageClassifier


def _trained(labelled_series):
    features = build_features(labelled_series(240))
    train, holdout = temporal_split(features, holdout_fraction=0.25)
    detector = AnomalyDetector(random_state=0).fit(train)
    classifier = TriageClassifier(random_state=0).fit(train)
    return detector, classifier, holdout


def test_models_round_trip_through_disk(tmp_path, labelled_series) -> None:
    detector, classifier, holdout = _trained(labelled_series)
    save_models(tmp_path, detector, classifier)
    loaded_detector, loaded_classifier = load_models(tmp_path)

    original = classifier.predict(holdout[holdout["is_anomaly"]])["cause"].to_numpy()
    restored = loaded_classifier.predict(holdout[holdout["is_anomaly"]])["cause"].to_numpy()
    assert (original == restored).all()

    pd.testing.assert_series_equal(
        pd.Series(detector.predict(holdout)), pd.Series(loaded_detector.predict(holdout))
    )


def test_log_training_run_returns_a_run_id_and_writes_the_store(tmp_path, labelled_series) -> None:
    detector, classifier, holdout = _trained(labelled_series)
    run_id = log_training_run(
        tracking_dir=tmp_path / "mlruns",
        detector=detector,
        classifier=classifier,
        triage_metrics=evaluate_triage(classifier, holdout),
        detector_metrics=evaluate_detector(detector, holdout),
        params={"baseline_window": 7, "holdout_fraction": 0.25},
    )
    assert isinstance(run_id, str) and run_id
    assert (tmp_path / "mlruns").exists()


def test_two_runs_get_distinct_ids(tmp_path, labelled_series) -> None:
    detector, classifier, holdout = _trained(labelled_series)
    common = dict(
        tracking_dir=tmp_path / "mlruns",
        detector=detector,
        classifier=classifier,
        triage_metrics=evaluate_triage(classifier, holdout),
        detector_metrics=evaluate_detector(detector, holdout),
        params={"baseline_window": 7},
    )
    assert log_training_run(**common) != log_training_run(**common)
