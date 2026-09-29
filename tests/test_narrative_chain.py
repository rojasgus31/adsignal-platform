import time
from datetime import date

import pytest

from adsignal.narrative.chain import AllProvidersFailed, NarrativeChain
from adsignal.narrative.providers.stub import StubProvider
from adsignal.narrative.types import NarrativeRequest

SLEEP_SECONDS = 0.02


class _SlowStubProvider:
    """A provider that takes a known amount of time to answer.

    Used only to pin down `NarrativeResult.latency_ms` against a real,
    known duration rather than the near-tautological `>= 0`.
    """

    def __init__(self, name: str = "slow") -> None:
        self.name = name

    def generate(self, request: NarrativeRequest) -> str:
        time.sleep(SLEEP_SECONDS)
        return "slow narrative"


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
    chain = NarrativeChain([_SlowStubProvider(name="a")])
    latency_ms = chain.generate(REQUEST).latency_ms
    assert latency_ms >= SLEEP_SECONDS * 1000


def test_an_empty_chain_raises_rather_than_returning_nothing() -> None:
    with pytest.raises(ValueError, match="at least one provider"):
        NarrativeChain([])


def test_duplicate_provider_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicated: dupe") as excinfo:
        NarrativeChain([StubProvider("a", name="dupe"), StubProvider("b", name="dupe")])
    assert "dupe" in str(excinfo.value)


def test_provider_order_is_preserved() -> None:
    chain = NarrativeChain(
        [
            StubProvider(fail_with=RuntimeError("x"), name="first"),
            StubProvider(fail_with=RuntimeError("y"), name="second"),
            StubProvider("third wins", name="third"),
        ]
    )
    assert chain.generate(REQUEST).provider == "third"


class _FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ClockBurningProvider:
    """A provider that succeeds, but only after consuming `seconds` of budget."""

    def __init__(self, clock: _FakeClock, seconds: float, name: str) -> None:
        self._clock = clock
        self._seconds = seconds
        self.name = name
        self.calls = 0

    def generate(self, request: NarrativeRequest) -> str:
        self.calls += 1
        self._clock.advance(self._seconds)
        return f"{self.name} narrative"


def test_a_provider_that_blows_the_total_budget_is_recorded_as_a_failure() -> None:
    """Its answer is discarded and its breaker counts it.

    Nothing else would ever open the breaker of a backend that is merely
    slow, so a permanently stalling provider would be retried in full, with
    its timeout, on every nightly run. The terminal fallback is exempt from
    the budget and still answers, so the caller gets placeholder text rather
    than nothing.
    """
    clock = _FakeClock()
    slow = _ClockBurningProvider(clock, seconds=30.0, name="slow")
    chain = NarrativeChain(
        [slow, StubProvider("fallback", name="fast")],
        failure_threshold=1,
        total_budget_seconds=10.0,
        clock=clock,
    )

    result = chain.generate(REQUEST)
    assert result.provider == "fast", "the terminal fallback must not be skipped"
    assert slow.calls == 1

    # The breaker opened on that recorded failure, so the next run skips the
    # slow provider without paying its cost again.
    result = chain.generate(REQUEST)
    assert result.provider == "fast"
    assert slow.calls == 1


def test_the_budget_is_spent_across_providers_not_per_provider() -> None:
    """Once the budget is gone, providers behind it are skipped.

    The terminal provider is the exception: it is the local fallback, so it
    is always entered and answers here.
    """
    clock = _FakeClock()
    first = _ClockBurningProvider(clock, seconds=12.0, name="first")
    second = _ClockBurningProvider(clock, seconds=12.0, name="second")
    third = _ClockBurningProvider(clock, seconds=12.0, name="third")
    chain = NarrativeChain([first, second, third], total_budget_seconds=10.0, clock=clock)

    result = chain.generate(REQUEST)

    assert (first.calls, second.calls, third.calls) == (1, 0, 1)
    assert result.provider == "third"


def test_a_provider_inside_the_budget_still_succeeds() -> None:
    clock = _FakeClock()
    quick = _ClockBurningProvider(clock, seconds=1.0, name="quick")
    chain = NarrativeChain([quick], total_budget_seconds=10.0, clock=clock)

    result = chain.generate(REQUEST)

    assert result.provider == "quick"
    assert result.latency_ms == pytest.approx(1000.0)


def test_a_non_positive_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="total_budget_seconds"):
        NarrativeChain([StubProvider("x", name="a")], total_budget_seconds=0.0)


def test_every_http_provider_failing_still_yields_the_stub_narrative() -> None:
    """The reason the stub exists: no keys, or every backend down, still renders.

    Regression test. The budget guard was once applied to the terminal
    provider too, so a chain slow enough to exhaust it skipped the free local
    fallback and raised instead of answering.
    """
    clock = _FakeClock()
    chain = NarrativeChain(
        [
            _ClockBurningProvider(clock, seconds=60.0, name="gemini"),
            StubProvider("placeholder narrative", name="stub"),
        ],
        total_budget_seconds=45.0,
        clock=clock,
    )
    result = chain.generate(REQUEST)
    assert result.provider == "stub"
    assert result.text == "placeholder narrative"


def test_a_fallback_answer_carries_what_it_fell_back_from() -> None:
    """ "The stub answered" and "gemini 404'd, then the stub answered" must differ.

    Without this the dashboard and `pipeline_runs` show the same thing for a
    deployment with no keys and for one whose paid provider has been failing
    every night — which is exactly what a retired model name looks like.
    """
    chain = NarrativeChain(
        [
            StubProvider(fail_with=RuntimeError("gemini returned 404"), name="gemini"),
            StubProvider("fallback text", name="stub"),
        ]
    )
    result = chain.generate(REQUEST)
    assert result.provider == "stub"
    assert result.skipped == ("gemini (RuntimeError)",)


def test_the_skipped_list_carries_no_provider_error_text() -> None:
    """Only names and exception TYPES, because a message can echo a key."""
    secret = "sk-super-secret-key-do-not-leak"
    chain = NarrativeChain(
        [
            StubProvider(fail_with=RuntimeError(f"failed with key={secret}"), name="gemini"),
            StubProvider("fallback text", name="stub"),
        ]
    )
    result = chain.generate(REQUEST)
    assert secret not in " ".join(result.skipped)


def test_a_first_provider_that_answers_reports_nothing_skipped() -> None:
    chain = NarrativeChain([StubProvider("first", name="a"), StubProvider("second", name="b")])
    assert chain.generate(REQUEST).skipped == ()
