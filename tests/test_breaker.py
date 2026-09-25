import pytest

from adsignal.narrative.breaker import CircuitBreaker


class FakeClock:
    """An injectable monotonic clock, so tests never sleep."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_a_new_breaker_is_closed() -> None:
    assert not CircuitBreaker().is_open()


def test_one_failure_below_the_threshold_keeps_it_closed() -> None:
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    assert not breaker.is_open()


def test_reaching_the_threshold_opens_it() -> None:
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open()


def test_breaker_closes_after_the_cooldown() -> None:
    """Review Focus 2: a breaker that never recovers turns a blip into an outage."""
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0, clock=clock)

    breaker.record_failure()
    assert breaker.is_open()

    clock.advance(29.0)
    assert breaker.is_open(), "it reopened before the cooldown elapsed"

    clock.advance(2.0)
    assert not breaker.is_open(), "it never recovered after the cooldown"


def test_success_resets_the_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    assert breaker.consecutive_failures == 0
    breaker.record_failure()
    assert not breaker.is_open()


def test_success_while_open_closes_it_immediately() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0, clock=clock)
    breaker.record_failure()
    assert breaker.is_open()
    breaker.record_success()
    assert not breaker.is_open()


def test_failure_after_recovery_reopens_it() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(11.0)
    assert not breaker.is_open()
    breaker.record_failure()
    assert breaker.is_open()


def test_rejects_a_nonsense_threshold() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        CircuitBreaker(failure_threshold=0)
