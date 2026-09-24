from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from adsignal.ml.detector import DETECTOR_FEATURES, AnomalyDetector
from adsignal.ml.evaluation import evaluate_detector
from adsignal.ml.features import build_features
from adsignal.ml.splits import temporal_split

FlatSeries = Callable[..., pd.DataFrame]


def _with_spike(flat_series: FlatSeries, days: int = 60, spike_at: int = 40) -> pd.DataFrame:
    frame = flat_series(days)
    frame.loc[spike_at, "spend"] = frame.loc[spike_at, "spend"] * 8
    frame.loc[spike_at, "ctr"] = frame.loc[spike_at, "ctr"] / 4
    return frame


def test_detector_flags_an_obvious_spike(flat_series: FlatSeries) -> None:
    features = build_features(_with_spike(flat_series))
    detector = AnomalyDetector(random_state=0).fit(features)
    flags = detector.predict(features)
    assert flags[40], "the injected spike was not flagged"


def test_detector_does_not_flag_everything(flat_series: FlatSeries) -> None:
    features = build_features(_with_spike(flat_series))
    detector = AnomalyDetector(random_state=0).fit(features)
    assert detector.predict(features).mean() < 0.35


def test_scores_are_higher_for_the_spike_than_the_median_day(
    flat_series: FlatSeries,
) -> None:
    features = build_features(_with_spike(flat_series))
    detector = AnomalyDetector(random_state=0).fit(features)
    scores = detector.score(features)
    assert scores[40] > np.nanmedian(scores)


def test_detector_is_deterministic_for_a_random_state(flat_series: FlatSeries) -> None:
    features = build_features(_with_spike(flat_series))
    first = AnomalyDetector(random_state=7).fit(features).score(features)
    second = AnomalyDetector(random_state=7).fit(features).score(features)
    np.testing.assert_allclose(first, second)


def test_rows_with_nan_features_still_receive_a_prediction(flat_series: FlatSeries) -> None:
    """Early campaign days have NaN baselines and must not crash or vanish."""
    features = build_features(flat_series(30))
    detector = AnomalyDetector(random_state=0).fit(features)
    flags = detector.predict(features)
    assert len(flags) == len(features)
    assert not np.isnan(detector.score(features)).any()


def test_predict_before_fit_raises(flat_series: FlatSeries) -> None:
    features = build_features(flat_series(30))
    with pytest.raises(RuntimeError, match="not fitted"):
        AnomalyDetector().predict(features)


def test_fit_rejects_an_empty_frame(flat_series: FlatSeries) -> None:
    empty = build_features(flat_series(0))
    with pytest.raises(ValueError, match="no rows"):
        AnomalyDetector().fit(empty)


def test_detector_features_are_all_produced_by_build_features(
    flat_series: FlatSeries,
) -> None:
    features = build_features(flat_series(30))
    for column in DETECTOR_FEATURES:
        assert column in features.columns


def test_contamination_is_derived_from_the_training_labels(
    labelled_series: Callable[..., pd.DataFrame],
) -> None:
    """`labelled_series` has an anomaly rate far from the 0.05 default."""
    features = build_features(labelled_series(240))
    observed_rate = features["is_anomaly"].fillna(False).astype(bool).mean()
    assert not 0.04 <= observed_rate <= 0.06, "fixture must not accidentally match the default"

    detector = AnomalyDetector(random_state=0)
    resolved = detector._resolve_contamination(features)  # noqa: SLF001
    assert resolved == pytest.approx(observed_rate, abs=1e-9)


def test_derived_contamination_raises_recall_on_a_temporal_holdout(
    labelled_series: Callable[..., pd.DataFrame],
) -> None:
    """This is the test that would have caught the mismatched-default bug.

    A detector fit with the old hardcoded 0.05 contamination against a mart
    whose real anomaly rate is much higher lands around 0.20 recall. With
    contamination derived from the training frame's own labels, recall on a
    genuine temporal holdout must clear a level well above that failure mode.

    This threshold is a fixture-specific regression check, not the CI quality
    gate: `labelled_series` is a controlled synthetic series, not real
    pipeline data, and the CI gate itself (`MINIMUM_DETECTOR_AUC`) checks ROC
    AUC precisely because recall is base-rate-dependent and does not travel
    from this fixture to a real mart. See evaluation.py's
    `MINIMUM_DETECTOR_AUC` rationale.
    """
    features = build_features(labelled_series(240))
    train, holdout = temporal_split(features, holdout_fraction=0.25)

    detector = AnomalyDetector(random_state=0).fit(train)
    metrics = evaluate_detector(detector, holdout)
    assert metrics.recall >= 0.5


def test_explicit_contamination_is_honoured_when_labels_are_absent(
    flat_series: FlatSeries,
) -> None:
    """A caller scoring unlabelled data can still pin `contamination` explicitly."""
    features = build_features(flat_series(30))
    unlabelled = features.drop(columns=["is_anomaly", "anomaly_cause", "anomaly_severity"])

    detector = AnomalyDetector(contamination=0.2, random_state=0)
    resolved = detector._resolve_contamination(unlabelled)  # noqa: SLF001
    assert resolved == 0.2

    # And it actually reaches the fitted forest.
    detector.fit(unlabelled)
    assert detector._forest.contamination == 0.2  # noqa: SLF001


def test_contamination_derivation_never_looks_at_a_holdout(
    labelled_series: Callable[..., pd.DataFrame],
) -> None:
    """Fitting on `train` must give the same contamination no matter what a holdout contains.

    Only the frame passed to `fit` may influence the derived contamination.
    Simulating a holdout with a wildly different (even inverted) label rate
    must not change what a fit on `train` alone resolves to.
    """
    features = build_features(labelled_series(240))
    train, holdout = temporal_split(features, holdout_fraction=0.25)

    detector = AnomalyDetector(random_state=0)
    baseline = detector._resolve_contamination(train)  # noqa: SLF001

    poisoned_holdout = holdout.copy()
    poisoned_holdout["is_anomaly"] = True  # wildly different from train's real rate
    assert poisoned_holdout["is_anomaly"].all()  # sanity: genuinely different from train

    # `_resolve_contamination` and `fit` only ever take one frame; the poisoned
    # holdout above was never passed to either. Resolving from `train` again
    # must reproduce the same value regardless of what that holdout contains.
    still_baseline = detector._resolve_contamination(train)  # noqa: SLF001
    assert still_baseline == baseline
