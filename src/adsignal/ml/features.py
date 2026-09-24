"""Feature engineering over the campaign-daily mart.

Pure domain logic: this module takes a DataFrame and returns a DataFrame. It
opens no connections, reads no files, and consults no environment.

Every rolling statistic is computed over a window that ENDS THE DAY BEFORE the
row it describes. Including the current day would let a model see the anomaly
it is supposed to predict - the model would score well offline and be useless
in production.
"""

import numpy as np
import pandas as pd

_RATE_COLUMNS = ("ctr", "cpc", "cpm", "cpa", "roas")

_PASSTHROUGH = (
    "impressions",
    "clicks",
    "conversions",
    "spend",
    "revenue",
    *_RATE_COLUMNS,
)

_BASELINE_METRICS = ("spend", "impressions", "clicks", "conversions", "ctr")

_ZSCORE_METRICS = ("spend", "ctr")

_IDENTIFIER_COLUMNS = ("event_date", "campaign_id")

_LABEL_COLUMNS = ("is_anomaly", "anomaly_cause", "anomaly_severity")

CATEGORICAL_COLUMNS: tuple[str, ...] = ("channel",)

# Explicit dtypes for the empty-frame path, mirroring `warehouse.reader`'s
# `_INT_COLUMNS`/`_FLOAT_COLUMNS`/`_STRING_COLUMNS` split. Without this, every
# declared column defaulted to float64 - wrong for 9 of them - so an empty
# date-window read produced a frame whose dtypes silently diverged from the
# populated path's (this is the same defect class fixed twice in
# `reader.py`, one module downstream). Any column not named here is a ratio,
# baseline or z-score and is genuinely float64.
_EMPTY_INT_COLUMNS = ("impressions", "clicks", "conversions")
# `.dt.dayofweek` yields int32, not the int64 of the mart's count columns.
_EMPTY_INT32_COLUMNS = ("day_of_week",)
# campaign_id/channel are non-nullable text; anomaly_cause is nullable text
# (NULL for every non-anomalous row). pandas' "str" dtype preserves missing
# values as NA, same as `reader._STRING_COLUMNS`.
_EMPTY_STRING_COLUMNS = ("campaign_id", "channel", "anomaly_cause")
_EMPTY_DATETIME_COLUMNS = ("event_date",)
_EMPTY_BOOL_COLUMNS = ("is_anomaly",)

FEATURE_COLUMNS: tuple[str, ...] = (
    *_PASSTHROUGH,
    "cvr",
    "spend_baseline",
    "spend_ratio",
    "impressions_baseline",
    "impressions_ratio",
    "clicks_baseline",
    "clicks_ratio",
    "conversions_baseline",
    "conversions_ratio",
    "ctr_baseline",
    "ctr_ratio",
    "spend_zscore",
    "ctr_zscore",
    "day_of_week",
    *CATEGORICAL_COLUMNS,
)


def build_features(
    frame: pd.DataFrame,
    baseline_window: int = 7,
    zscore_window: int = 28,
) -> pd.DataFrame:
    """Return one feature row per input row, ordered as the input was.

    Rolling features are NaN until a campaign has `baseline_window` prior days.
    That is correct: an unknown baseline must read as unknown, not as zero.

    Does not mutate `frame`. Every column in `FEATURE_COLUMNS` is present in
    the result, even when `frame` is empty.
    """
    if frame.empty:
        return _empty_frame(frame)

    working = frame.copy(deep=True)
    working = working.sort_values(list(_IDENTIFIER_COLUMNS))

    _add_conversion_rate(working)
    _add_baselines_and_ratios(working, baseline_window)
    _add_zscores(working, baseline_window, zscore_window)
    working["day_of_week"] = working["event_date"].dt.dayofweek

    keep = [*_IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]
    if "is_anomaly" in working.columns:
        keep.extend(_LABEL_COLUMNS)

    return working.loc[:, keep].reindex(frame.index)


def _add_conversion_rate(working: pd.DataFrame) -> None:
    """Conversion rate is not in the mart; clicks of 0 leave it undefined."""
    clicks = working["clicks"].replace(0, np.nan)
    working["cvr"] = working["conversions"] / clicks


def _add_baselines_and_ratios(working: pd.DataFrame, baseline_window: int) -> None:
    """For each baseline metric, add `<metric>_baseline` and `<metric>_ratio`.

    The baseline is the mean of the `baseline_window` days STRICTLY BEFORE the
    row's day, grouped by campaign so history never bleeds across campaigns.
    """
    campaign_ids = working["campaign_id"]
    for metric in _BASELINE_METRICS:
        prior = working.groupby("campaign_id", sort=False)[metric].shift(1)
        baseline = (
            prior.groupby(campaign_ids, sort=False)
            .rolling(window=baseline_window, min_periods=baseline_window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        working[f"{metric}_baseline"] = baseline
        working[f"{metric}_ratio"] = working[metric] / baseline.replace(0, np.nan)


def _add_zscores(working: pd.DataFrame, baseline_window: int, zscore_window: int) -> None:
    """For each z-score metric, add `<metric>_zscore` from prior-day rolling mean/std."""
    campaign_ids = working["campaign_id"]
    for metric in _ZSCORE_METRICS:
        prior = working.groupby("campaign_id", sort=False)[metric].shift(1)
        rolling = prior.groupby(campaign_ids, sort=False).rolling(
            window=zscore_window, min_periods=baseline_window
        )
        mean = rolling.mean().reset_index(level=0, drop=True)
        std = rolling.std().reset_index(level=0, drop=True)
        working[f"{metric}_zscore"] = (working[metric] - mean) / std.replace(0, np.nan)


def _empty_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an empty frame that still carries every declared column.

    Dtypes are made explicit rather than left to default to float64, so this
    path agrees with the populated path's dtypes column-for-column - see
    `test_empty_frame_dtypes_match_the_populated_path`.
    """
    columns = [*_IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]
    if "is_anomaly" in frame.columns:
        columns.extend(_LABEL_COLUMNS)

    def _dtype(column: str) -> str:
        if column in _EMPTY_INT_COLUMNS:
            return "int64"
        if column in _EMPTY_INT32_COLUMNS:
            return "int32"
        if column in _EMPTY_STRING_COLUMNS:
            return "str"
        if column in _EMPTY_DATETIME_COLUMNS:
            return "datetime64[ns]"
        if column in _EMPTY_BOOL_COLUMNS:
            return "bool"
        return "float64"

    return pd.DataFrame({column: pd.Series(dtype=_dtype(column)) for column in columns})
