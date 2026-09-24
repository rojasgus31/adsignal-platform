import dataclasses
from datetime import date

import pytest

from adsignal.generator.models import (
    ANOMALY_CAUSES,
    CHANNELS,
    AnomalyLabel,
    Campaign,
    EventRow,
)


def test_channels_are_the_five_documented_ones() -> None:
    assert CHANNELS == ("search", "social", "display", "video", "native")


def test_anomaly_causes_are_the_five_documented_ones() -> None:
    assert ANOMALY_CAUSES == (
        "budget_pacing",
        "creative_fatigue",
        "audience_saturation",
        "tracking_error",
        "competitive_pressure",
    )


def test_event_row_is_frozen() -> None:
    row = EventRow(
        event_date=date(2024, 1, 1),
        campaign_id="cmp_0001",
        creative_id="crv_0001",
        channel="search",
        impressions=1000,
        clicks=20,
        conversions=2,
        spend=15.0,
        revenue=60.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.clicks = 5  # type: ignore[misc]


def test_campaign_rejects_unknown_channel() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        Campaign(
            id="cmp_0001",
            advertiser_id="adv_0001",
            name="Q1 Brand",
            channel="carrier_pigeon",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            daily_budget=500.0,
        )


def test_anomaly_label_rejects_unknown_cause() -> None:
    with pytest.raises(ValueError, match="unknown cause"):
        AnomalyLabel(
            event_date=date(2024, 1, 1),
            campaign_id="cmp_0001",
            cause="sunspots",
            severity=3,
        )


def test_campaign_rejects_end_date_before_start_date() -> None:
    with pytest.raises(ValueError, match="end_date precedes start_date"):
        Campaign(
            id="cmp_0001",
            advertiser_id="adv_0001",
            name="Q1 Brand",
            channel="search",
            start_date=date(2024, 3, 31),
            end_date=date(2024, 1, 1),
            daily_budget=500.0,
        )


def test_anomaly_label_rejects_severity_too_low() -> None:
    with pytest.raises(ValueError, match="severity must be between 1 and 5"):
        AnomalyLabel(
            event_date=date(2024, 1, 1),
            campaign_id="cmp_0001",
            cause="budget_pacing",
            severity=0,
        )


def test_anomaly_label_rejects_severity_too_high() -> None:
    with pytest.raises(ValueError, match="severity must be between 1 and 5"):
        AnomalyLabel(
            event_date=date(2024, 1, 1),
            campaign_id="cmp_0001",
            cause="budget_pacing",
            severity=6,
        )
