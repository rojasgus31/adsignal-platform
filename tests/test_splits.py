import pandas as pd
import pytest

from adsignal.ml.splits import split_date, temporal_split


def _frame(days: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame({"event_date": dates, "value": range(days)})


def test_split_is_temporal_not_random() -> None:
    """Review Focus 2: every training date must precede every holdout date.

    A random split would train on Thursday to predict Wednesday.
    """
    train, holdout = temporal_split(_frame(), holdout_fraction=0.25)
    assert len(train) > 0 and len(holdout) > 0
    assert train["event_date"].max() < holdout["event_date"].min()


def test_split_covers_every_row_exactly_once() -> None:
    frame = _frame()
    train, holdout = temporal_split(frame, holdout_fraction=0.25)
    assert len(train) + len(holdout) == len(frame)
    combined = pd.concat([train, holdout]).sort_values("value").reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, frame.sort_values("value").reset_index(drop=True))


def test_holdout_fraction_is_approximately_honored() -> None:
    train, holdout = temporal_split(_frame(100), holdout_fraction=0.25)
    assert 20 <= len(holdout) <= 30


def test_same_date_never_straddles_the_split() -> None:
    """All rows sharing a date must land on the same side, or the split leaks."""
    dates = pd.to_datetime(["2024-01-01"] * 10 + ["2024-01-02"] * 10 + ["2024-01-03"] * 10)
    frame = pd.DataFrame({"event_date": dates, "value": range(30)})
    train, holdout = temporal_split(frame, holdout_fraction=0.34)
    overlap = set(train["event_date"]) & set(holdout["event_date"])
    assert not overlap, f"dates on both sides of the split: {overlap}"


def test_rejects_frame_with_too_few_distinct_dates() -> None:
    frame = pd.DataFrame({"event_date": pd.to_datetime(["2024-01-01"] * 5), "value": range(5)})
    with pytest.raises(ValueError, match="at least 2 distinct dates"):
        temporal_split(frame, holdout_fraction=0.25)


def test_split_date_is_the_first_holdout_date() -> None:
    frame = _frame(100)
    boundary = split_date(frame, holdout_fraction=0.25)
    _, holdout = temporal_split(frame, holdout_fraction=0.25)
    assert holdout["event_date"].min() == boundary
