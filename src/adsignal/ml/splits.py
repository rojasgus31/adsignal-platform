"""Splitting a campaign-day series into training and holdout sets.

The split is by date, never random. A random split over a time series trains on
the future to predict the past, which inflates every offline metric and tells
you nothing about production behaviour.
"""

import pandas as pd


def split_date(
    frame: pd.DataFrame,
    holdout_fraction: float = 0.25,
    date_column: str = "event_date",
) -> pd.Timestamp:
    """Return the first date belonging to the holdout set."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")

    distinct = pd.Series(frame[date_column].unique()).sort_values()
    if len(distinct) < 2:
        raise ValueError("a temporal split needs at least 2 distinct dates")

    holdout_days = max(1, min(len(distinct) - 1, round(len(distinct) * holdout_fraction)))
    return pd.Timestamp(distinct.iloc[len(distinct) - holdout_days])


def temporal_split(
    frame: pd.DataFrame,
    holdout_fraction: float = 0.25,
    date_column: str = "event_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, holdout) so every training date precedes every holdout date.

    Rows sharing a date always land on the same side.
    """
    boundary = split_date(frame, holdout_fraction, date_column)
    train = frame[frame[date_column] < boundary].copy()
    holdout = frame[frame[date_column] >= boundary].copy()
    return train, holdout
