"""Domain types for synthetic ad-campaign data.

These are pure data structures. Nothing here performs I/O or imports from
the warehouse or pipeline layers.
"""

from dataclasses import dataclass
from datetime import date

CHANNELS: tuple[str, ...] = ("search", "social", "display", "video", "native")

ANOMALY_CAUSES: tuple[str, ...] = (
    "budget_pacing",
    "creative_fatigue",
    "audience_saturation",
    "tracking_error",
    "competitive_pressure",
)

CREATIVE_FORMATS: tuple[str, ...] = ("banner", "video_15s", "video_30s", "carousel", "text")

VERTICALS: tuple[str, ...] = ("retail", "finance", "travel", "auto", "cpg", "telecom")


@dataclass(frozen=True, slots=True)
class Advertiser:
    """A brand buying media."""

    id: str
    name: str
    vertical: str


@dataclass(frozen=True, slots=True)
class Campaign:
    """A media buy for one advertiser on one channel over a date range."""

    id: str
    advertiser_id: str
    name: str
    channel: str
    start_date: date
    end_date: date
    daily_budget: float

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(f"unknown channel: {self.channel}")
        if self.end_date < self.start_date:
            raise ValueError("end_date precedes start_date")


@dataclass(frozen=True, slots=True)
class Creative:
    """A single ad execution belonging to a campaign."""

    id: str
    campaign_id: str
    headline: str
    format: str


@dataclass(frozen=True, slots=True)
class Catalog:
    """The full set of dimension entities for one generated dataset."""

    advertisers: tuple[Advertiser, ...]
    campaigns: tuple[Campaign, ...]
    creatives: tuple[Creative, ...]


@dataclass(frozen=True, slots=True)
class EventRow:
    """One creative's aggregated delivery for one day."""

    event_date: date
    campaign_id: str
    creative_id: str
    channel: str
    impressions: int
    clicks: int
    conversions: int
    spend: float
    revenue: float


@dataclass(frozen=True, slots=True)
class AnomalyLabel:
    """Ground truth for one injected anomaly, keyed by campaign-day."""

    event_date: date
    campaign_id: str
    cause: str
    severity: int

    def __post_init__(self) -> None:
        if self.cause not in ANOMALY_CAUSES:
            raise ValueError(f"unknown cause: {self.cause}")
        if not 1 <= self.severity <= 5:
            raise ValueError("severity must be between 1 and 5")
