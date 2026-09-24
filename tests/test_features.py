import numpy as np
import pandas as pd
import pytest

from adsignal.ml.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_features


def test_output_has_the_declared_feature_columns(flat_series) -> None:
    features = build_features(flat_series())
    for column in FEATURE_COLUMNS:
        assert column in features.columns, f"missing feature: {column}"


def test_one_row_per_input_row(flat_series) -> None:
    frame = flat_series(40)
    assert len(build_features(frame)) == len(frame)


def test_rolling_baseline_excludes_current_day(flat_series) -> None:
    """Review Focus 1: the single most damaging bug in an ML pipeline.

    A baseline that includes today lets the model see the anomaly it must
    predict. Spike one day's spend 10x: the baseline for THAT day must be
    unmoved, so the ratio feature actually registers the spike.
    """
    frame = flat_series(40, spend=100.0)
    spike_index = 20
    frame.loc[spike_index, "spend"] = 1000.0

    features = build_features(frame, baseline_window=7)
    spike_row = features.iloc[spike_index]

    assert spike_row["spend_baseline"] == pytest.approx(100.0, rel=1e-6), (
        "the baseline moved with the spike, so it includes the current day"
    )
    assert spike_row["spend_ratio"] == pytest.approx(10.0, rel=1e-6)


def test_null_rates_stay_nan_and_are_not_imputed(flat_series) -> None:
    """Review Focus 3: a NULL CPA means undefined, not zero.

    Imputing 0 teaches the model that a zero-conversion day converts for free.
    """
    frame = flat_series(40)
    frame.loc[10, "conversions"] = 0
    frame.loc[10, "cpa"] = np.nan
    frame.loc[10, "roas"] = np.nan

    features = build_features(frame)
    assert pd.isna(features.loc[10, "cpa"])
    assert pd.isna(features.loc[10, "roas"])
    assert (features["cpa"] == 0).sum() == 0, "a NaN CPA was silently filled with 0"


def test_short_campaign_yields_nan_not_garbage(flat_series) -> None:
    """Review Focus 4: a 3-day campaign cannot have a 7-day baseline."""
    frame = flat_series(3)
    features = build_features(frame, baseline_window=7)

    assert len(features) == 3
    assert features["spend_baseline"].isna().all(), (
        "a baseline was produced from fewer days than the window"
    )
    assert np.isfinite(features["spend"]).all(), "raw passthrough features should still be present"


def test_first_day_of_a_campaign_has_no_baseline(flat_series) -> None:
    features = build_features(flat_series(40))
    assert pd.isna(features.iloc[0]["spend_baseline"])
    assert pd.isna(features.iloc[0]["spend_ratio"])


def test_baselines_do_not_bleed_across_campaigns(flat_series) -> None:
    """Campaign B's first day must not inherit campaign A's history."""
    first = flat_series(20, spend=100.0)
    second = flat_series(20, spend=900.0)
    second["campaign_id"] = "cmp_b"
    frame = pd.concat([first, second], ignore_index=True)

    features = build_features(frame, baseline_window=7)
    second_start = features[features["campaign_id"] == "cmp_b"].iloc[0]
    assert pd.isna(second_start["spend_baseline"])


def test_day_of_week_is_present_and_in_range(flat_series) -> None:
    features = build_features(flat_series(40))
    assert features["day_of_week"].between(0, 6).all()


def test_channel_is_carried_through_as_a_categorical(flat_series) -> None:
    features = build_features(flat_series(40))
    for column in CATEGORICAL_COLUMNS:
        assert column in features.columns
    assert features["channel"].iloc[0] == "search"


def test_build_features_does_not_mutate_its_input(flat_series) -> None:
    frame = flat_series(40)
    before = frame.copy(deep=True)
    build_features(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_empty_frame_returns_empty_with_feature_columns(flat_series) -> None:
    empty = flat_series(0)
    features = build_features(empty)
    assert len(features) == 0
    for column in FEATURE_COLUMNS:
        assert column in features.columns


def test_empty_frame_dtypes_match_the_populated_path(flat_series) -> None:
    """I1: the empty path must not silently diverge from the populated one.

    A frame with no rows to infer from previously defaulted every column to
    float64, which is wrong for `event_date` (datetime), `campaign_id`/
    `channel`/`anomaly_cause` (text), `impressions`/`clicks`/`conversions`/
    `day_of_week` (int) and `is_anomaly` (bool) - the same defect class fixed
    twice in `warehouse.reader` during Task 1, left standing here.
    """
    # `flat_series` itself doesn't emulate `warehouse.reader`'s explicit casts
    # (real callers always go through the reader first), so apply the two
    # casts that matter here before comparing - otherwise the comparison
    # would be checking `flat_series`'s incidental construction, not the
    # populated-vs-empty dtype contract `build_features` promises its callers.
    populated_input = flat_series(5)
    populated_input["event_date"] = populated_input["event_date"].astype("datetime64[ns]")
    populated_input["anomaly_cause"] = populated_input["anomaly_cause"].astype("str")

    populated = build_features(populated_input)
    empty = build_features(flat_series(0))
    assert empty.dtypes.equals(populated.dtypes)
