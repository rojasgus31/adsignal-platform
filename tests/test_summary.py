"""The terminal fallback must be useful, not a placeholder."""

from datetime import date

from adsignal.narrative.summary import summarize
from adsignal.narrative.types import NarrativeRequest


def _request(**overrides) -> NarrativeRequest:
    base = {
        "as_of": date(2025, 12, 31),
        "campaign_count": 96,
        "anomaly_count": 1918,
        "top_causes": ("creative_fatigue", "budget_pacing", "tracking_error"),
        "escalated_count": 14,
    }
    return NarrativeRequest(**{**base, **overrides})


def test_it_never_says_placeholder() -> None:
    """The exact failure this task exists to remove."""
    text = summarize(_request()).lower()
    assert "placeholder" not in text
    assert "not configured" not in text


def test_it_names_the_real_figures() -> None:
    text = summarize(_request())
    assert "96" in text
    assert "1,918" in text or "1918" in text
    assert "creative_fatigue" in text or "creative fatigue" in text


def test_a_day_with_no_data_says_so() -> None:
    """Review Focus 5. Never a crash, never an empty string."""
    text = summarize(_request(campaign_count=0, anomaly_count=0, top_causes=(), escalated_count=0))
    assert text.strip()
    assert "no" in text.lower()
    assert "placeholder" not in text.lower()


def test_a_clean_day_reads_as_clean() -> None:
    text = summarize(_request(anomaly_count=0, top_causes=(), escalated_count=0))
    assert text.strip()
    assert "1,918" not in text


def test_it_is_deterministic() -> None:
    """Two calls with the same request must be byte-identical."""
    assert summarize(_request()) == summarize(_request())


def test_escalations_are_mentioned_when_present() -> None:
    assert "14" in summarize(_request(escalated_count=14))


def test_no_escalations_is_not_silently_omitted() -> None:
    text = summarize(_request(escalated_count=0))
    assert text.strip()
