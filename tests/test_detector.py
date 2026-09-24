from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from adsignal.ml.detector import DETECTOR_FEATURES, AnomalyDetector
from adsignal.ml.features import build_features

FlatSeries = Callable[..., pd.DataFrame]


def _with_spike(
    flat_series: FlatSeries, days: int = 60, spike_at: int = 40
) -> pd.DataFrame:
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
