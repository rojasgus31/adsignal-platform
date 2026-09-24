"""Injection of labelled anomalies into the event stream.

Each anomaly covers a contiguous run of days for one campaign and carries a
ground-truth cause. A campaign-day can belong to at most one anomaly, so the
resulting label set is unambiguous for supervised training.
"""

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta

from adsignal.generator.models import ANOMALY_CAUSES, AnomalyLabel, EventRow


@dataclass(frozen=True, slots=True)
class _Effect:
    """Multipliers applied to a row for one anomaly cause."""

    spend: float
    impressions: float
    clicks: float
    conversions: float


_EFFECTS: dict[str, _Effect] = {
    "budget_pacing": _Effect(spend=2.4, impressions=2.2, clicks=2.0, conversions=1.1),
    "creative_fatigue": _Effect(spend=1.0, impressions=1.0, clicks=0.45, conversions=0.40),
    "audience_saturation": _Effect(spend=1.3, impressions=1.4, clicks=0.8, conversions=0.35),
    "tracking_error": _Effect(spend=1.0, impressions=1.0, clicks=1.0, conversions=0.0),
    "competitive_pressure": _Effect(spend=1.6, impressions=0.7, clicks=0.65, conversions=0.6),
}


class AnomalyInjector:
    """Perturbs an event stream and records ground truth.

    Labels are appended as the stream is consumed, one per campaign-day that
    actually had a row processed for it — never for a day with no row. Read
    `.labels` only after the iterator returned by `.rows()` has been
    exhausted. Windows are refused if they would overlap an existing window
    for the same campaign, so at most one label is ever produced per
    campaign-day, regardless of the order rows for that campaign arrive in.
    """

    def __init__(
        self,
        *,
        seed: int,
        rate: float = 0.01,
        min_duration: int = 2,
        max_duration: int = 7,
        forced_cause: str | None = None,
    ) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be between 0.0 and 1.0")
        if forced_cause is not None and forced_cause not in ANOMALY_CAUSES:
            raise ValueError(f"unknown cause: {forced_cause}")

        self._seed = seed
        self._rate = rate
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._forced_cause = forced_cause
        self.labels: list[AnomalyLabel] = []
        self._windows: dict[str, list[tuple[date, date, str, int]]] = {}
        self._labelled: set[tuple[str, date]] = set()

    def rows(self, source: Iterable[EventRow]) -> Iterator[EventRow]:
        """Yield the source rows, perturbing those inside an anomaly window.

        A label is appended only when a row for that campaign-day is actually
        processed, guarded by `_labelled` so that multiple creatives sharing a
        campaign-day still produce exactly one label.
        """
        for row in source:
            match = self._window_for(row.campaign_id, row.event_date)
            if match is None:
                yield row
                continue
            _, _, cause, severity = match
            key = (row.campaign_id, row.event_date)
            if key not in self._labelled:
                self._labelled.add(key)
                self.labels.append(
                    AnomalyLabel(
                        event_date=row.event_date,
                        campaign_id=row.campaign_id,
                        cause=cause,
                        severity=severity,
                    )
                )
            yield _apply(row, _EFFECTS[cause], severity)

    def _window_for(self, campaign_id: str, event_date: date) -> tuple[date, date, str, int] | None:
        """Return the anomaly window covering this campaign-day, deciding lazily.

        A new window is refused if it would overlap any existing window for
        this campaign, regardless of the order rows arrive in. This is what
        guarantees at most one label per campaign-day even for an
        out-of-order input stream: without this check, a later out-of-order
        row could open a second window over days an earlier-opened window
        already covers.
        """
        for window in self._windows.get(campaign_id, ()):
            start, end, _, _ = window
            if start <= event_date <= end:
                return window

        if self._rate <= 0.0:
            return None

        # random.Random only accepts None/int/float/str/bytes/bytearray, so the
        # seed components are joined into a single string (mirrors events.py).
        row_seed = f"{self._seed}:{campaign_id}:{event_date.toordinal()}"
        rng = random.Random(row_seed)
        if rng.random() >= self._rate:
            return None

        cause = self._forced_cause or rng.choice(ANOMALY_CAUSES)
        severity = rng.randint(1, 5)
        duration = rng.randint(self._min_duration, self._max_duration)
        end = event_date + timedelta(days=duration - 1)

        if self._overlaps_existing(campaign_id, event_date, end):
            return None

        window = (event_date, end, cause, severity)
        self._windows.setdefault(campaign_id, []).append(window)
        return window

    def _overlaps_existing(self, campaign_id: str, start: date, end: date) -> bool:
        """Return True if [start, end] overlaps any existing window for this campaign."""
        for existing_start, existing_end, _, _ in self._windows.get(campaign_id, ()):
            if start <= existing_end and existing_start <= end:
                return True
        return False


def _apply(row: EventRow, effect: _Effect, severity: int) -> EventRow:
    """Scale a row's metrics, keeping impressions >= clicks >= conversions."""
    strength = 0.4 + 0.15 * severity

    def scale(value: float, multiplier: float) -> float:
        return value * (1.0 + (multiplier - 1.0) * strength)

    impressions = max(0, round(scale(row.impressions, effect.impressions)))
    clicks = min(impressions, max(0, round(scale(row.clicks, effect.clicks))))
    if effect.conversions == 0.0:
        # A multiplier of exactly 0.0 (e.g. tracking_error) means conversions
        # are completely lost regardless of severity, not merely scaled down
        # by it — a broken tracking pixel does not fire "a little less".
        conversions = 0
    else:
        conversions = min(clicks, max(0, round(scale(row.conversions, effect.conversions))))
    # scale(0, mult) is always 0, so conversions == 0 whenever row.conversions
    # == 0; reaching the else branch therefore implies row.conversions > 0,
    # making a divisor guard unreachable and unnecessary.
    revenue = 0.0 if conversions == 0 else row.revenue * (conversions / row.conversions)

    return EventRow(
        event_date=row.event_date,
        campaign_id=row.campaign_id,
        creative_id=row.creative_id,
        channel=row.channel,
        impressions=impressions,
        clicks=clicks,
        conversions=conversions,
        spend=round(max(0.0, scale(row.spend, effect.spend)), 4),
        revenue=round(max(0.0, revenue), 4),
    )
