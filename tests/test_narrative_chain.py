from datetime import date

import pytest

from adsignal.narrative.chain import AllProvidersFailed, NarrativeChain
from adsignal.narrative.providers.stub import StubProvider
from adsignal.narrative.types import NarrativeRequest

REQUEST = NarrativeRequest(
    as_of=date(2024, 6, 1),
    campaign_count=12,
    anomaly_count=3,
    top_causes=("budget_pacing", "creative_fatigue"),
    escalated_count=1,
)


def test_the_first_healthy_provider_wins() -> None:
    chain = NarrativeChain([StubProvider("first", name="a"), StubProvider("second", name="b")])
    result = chain.generate(REQUEST)
    assert result.text == "first"
    assert result.provider == "a"


def test_it_falls_through_to_the_next_provider_on_failure() -> None:
    chain = NarrativeChain(
        [
            StubProvider(fail_with=RuntimeError("upstream 503"), name="broken"),
            StubProvider("recovered", name="healthy"),
        ]
    )
    result = chain.generate(REQUEST)
    assert result.text == "recovered"
    assert result.provider == "healthy"


def test_every_provider_failing_raises() -> None:
    chain = NarrativeChain(
        [
            StubProvider(fail_with=RuntimeError("down"), name="a"),
            StubProvider(fail_with=RuntimeError("also down"), name="b"),
        ]
    )
    with pytest.raises(AllProvidersFailed) as excinfo:
        chain.generate(REQUEST)
    assert "a" in str(excinfo.value) and "b" in str(excinfo.value)


def test_a_failing_provider_is_skipped_once_its_breaker_opens() -> None:
    broken = StubProvider(fail_with=RuntimeError("down"), name="broken")
    healthy = StubProvider("ok", name="healthy")
    chain = NarrativeChain([broken, healthy], failure_threshold=1)

    chain.generate(REQUEST)
    calls_after_first = broken.call_count
    chain.generate(REQUEST)

    assert broken.call_count == calls_after_first, "the open breaker did not skip the provider"


def test_result_records_which_provider_answered() -> None:
    chain = NarrativeChain([StubProvider("text", name="gemini-stub")])
    assert chain.generate(REQUEST).provider == "gemini-stub"


def test_result_records_a_latency() -> None:
    chain = NarrativeChain([StubProvider("text", name="a")])
    assert chain.generate(REQUEST).latency_ms >= 0


def test_an_empty_chain_raises_rather_than_returning_nothing() -> None:
    with pytest.raises(ValueError, match="at least one provider"):
        NarrativeChain([])


def test_provider_order_is_preserved() -> None:
    chain = NarrativeChain(
        [
            StubProvider(fail_with=RuntimeError("x"), name="first"),
            StubProvider(fail_with=RuntimeError("y"), name="second"),
            StubProvider("third wins", name="third"),
        ]
    )
    assert chain.generate(REQUEST).provider == "third"
