"""Assembling one campaign-day's model-ready feature row from the mart.

`POST /score` needs the same feature row the nightly batch scores, for a
single `(campaign_id, event_date)`. That row cannot be computed from that
day's own metrics: every rolling column in
`adsignal.ml.features.FEATURE_COLUMNS` — `spend_baseline`, `ctr_ratio`,
`spend_zscore` and the rest — is a statistic over a window that ENDS THE DAY
BEFORE, which is the leakage-prevention design `features.py` documents at
length. So this module reads the campaign's recent history, builds features
over it, and returns only the requested day's row.

Two consequences follow, and both are this module's responsibility because
only the layer that read the history knows the answer:

* A campaign-day with fewer than `DEFAULT_BASELINE_WINDOW` prior days has NaN
  for every baseline and ratio. Scoring it would produce a confident,
  meaningless number, so this module refuses and names the shortfall instead.
* Counting rows is necessary but not sufficient. A day can have its seven
  prior days and still be NaN in every feature the detector reads — zero
  baselines and zero-variance windows do it, and seven days of zero delivery
  produces both. The detector neutral-fills NaN, so such a row scores as
  exactly average no matter what it contains. `_carries_no_evidence` catches
  it, with a reason distinct from the shortfall one so a caller can tell a
  too-young campaign from a dead one.
* The read is bounded to `DEFAULT_ZSCORE_WINDOW` prior days, the widest
  window `build_features` uses. That is not an approximation: the rolling
  statistics for the last row of a window are computed from exactly those
  prior rows, so this reproduces the batch path's values while reading ~29
  rows instead of a campaign's entire history.

`api/` calls this and issues no SQL of its own.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from adsignal.ml.detector import DETECTOR_FEATURES
from adsignal.ml.features import DEFAULT_BASELINE_WINDOW, DEFAULT_ZSCORE_WINDOW, build_features
from adsignal.warehouse.loader import ensure_schema
from adsignal.warehouse.reader import read_campaign_window


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """One campaign-day's feature row, ready to hand to a fitted model."""

    campaign_id: str
    event_date: date
    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class NoFeatureRow:
    """Why a campaign-day has no scoreable feature row.

    Carried as a value rather than raised: neither an unknown campaign-day nor
    a too-young one is an error in this process, and the caller turns both
    into the same refusal. `reason` is written for the caller to surface
    verbatim, so it names the shortfall in the caller's own terms and carries
    nothing but the campaign-day asked for and two counts.
    """

    reason: str


def read_feature_row(
    dsn: str,
    campaign_id: str,
    event_date: date,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
) -> FeatureRow | NoFeatureRow:
    """Return the feature row for one campaign-day, or why there isn't one.

    `ensure_schema` runs first so a database the pipeline has never run
    against answers "no such campaign-day" rather than raising an undefined-
    table error.
    """
    ensure_schema(dsn)
    history = read_campaign_window(dsn, campaign_id, event_date, days=DEFAULT_ZSCORE_WINDOW + 1)

    if not _ends_on(history, event_date):
        return NoFeatureRow(
            reason=f"no mart row for campaign {campaign_id!r} on {event_date.isoformat()}"
        )

    prior_days = len(history) - 1
    if prior_days < baseline_window:
        return NoFeatureRow(
            reason=(
                f"campaign {campaign_id!r} has only {prior_days} day(s) of history before "
                f"{event_date.isoformat()}; scoring needs {baseline_window}, because every "
                "baseline and ratio feature is a statistic over the preceding "
                f"{baseline_window} days and is undefined until then"
            )
        )

    row = build_features(history, baseline_window=baseline_window).tail(1)
    if _carries_no_evidence(row):
        return NoFeatureRow(
            reason=(
                f"campaign {campaign_id!r} on {event_date.isoformat()} has {len(history) - 1} "
                "day(s) of history but every feature the detector reads is undefined "
                f"({', '.join(DETECTOR_FEATURES)}); a zero baseline, a zero-variance window "
                "or zero delivery leaves nothing to compare this day against"
            )
        )

    return FeatureRow(campaign_id=campaign_id, event_date=event_date, frame=row)


def _carries_no_evidence(row: pd.DataFrame) -> bool:
    """True when every feature the detector reads is NaN on this row.

    The row-count guard above is necessary but not sufficient. A campaign-day
    can have its seven prior days and still be undefined in every detector
    feature: a `*_ratio` is NaN when the baseline is zero
    (`_add_baselines_and_ratios` replaces a zero baseline with NaN rather than
    dividing by it), and a `*_zscore` is NaN when the window's standard
    deviation is zero. Seven days of zero delivery produces exactly that.

    `AnomalyDetector._matrix` then fills each NaN with its column's neutral
    value and subtracts that neutral, so a row that is NaN throughout reaches
    the forest as the origin and scores as perfectly average — the same
    confident, meaningless number the history guard exists to refuse,
    arriving by a different door.

    The threshold is *every* feature, not any. A single NaN ratio is a normal
    and informative state — a campaign with no conversions in its baseline
    week has a NaN `conversions_ratio` and six other features still carrying
    real deviation — and the detector's neutral-fill policy handles that
    correctly and identically to the batch path. Only the all-NaN row is
    scored independently of its own data, and only it is refused; anything
    stricter would refuse legitimate days and make `/score` disagree with the
    stored batch verdict for no gain.
    """
    present = [column for column in DETECTOR_FEATURES if column in row.columns]
    return bool(present) and bool(row.loc[:, present].isna().all(axis=None))


def _ends_on(history: pd.DataFrame, event_date: date) -> bool:
    """True when `history`'s last (newest) row is the requested day itself.

    The window read is `event_date <= %s`, so a campaign with a gap around the
    requested day still returns rows; the day asked about simply is not one of
    them, and scoring the day before it instead would answer a question nobody
    asked.
    """
    if history.empty:
        return False
    newest: pd.Timestamp = history["event_date"].iloc[-1]
    return bool(newest.date() == event_date)
