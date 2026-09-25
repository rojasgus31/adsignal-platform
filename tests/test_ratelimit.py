import pytest

from adsignal.api.ratelimit import TokenBucket


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_requests_within_capacity_are_allowed() -> None:
    bucket = TokenBucket(capacity=3, refill_per_second=0.0, clock=FakeClock())
    assert all(bucket.allow("1.2.3.4") for _ in range(3))


def test_the_request_past_capacity_is_refused() -> None:
    bucket = TokenBucket(capacity=2, refill_per_second=0.0, clock=FakeClock())
    bucket.allow("ip")
    bucket.allow("ip")
    assert not bucket.allow("ip")


def test_separate_keys_have_separate_budgets() -> None:
    bucket = TokenBucket(capacity=1, refill_per_second=0.0, clock=FakeClock())
    assert bucket.allow("a")
    assert bucket.allow("b")
    assert not bucket.allow("a")


def test_tokens_refill_over_time() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, refill_per_second=1.0, clock=clock)
    assert bucket.allow("ip")
    assert not bucket.allow("ip")
    clock.advance(1.5)
    assert bucket.allow("ip")


def test_refill_does_not_exceed_capacity() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, refill_per_second=1.0, clock=clock)
    clock.advance(1000.0)
    assert bucket.allow("ip")
    assert bucket.allow("ip")
    assert not bucket.allow("ip")


def test_rejects_a_nonsense_capacity() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        TokenBucket(capacity=0, refill_per_second=1.0)


def test_a_negative_refill_rate_is_rejected() -> None:
    """A negative rate would drive tokens below zero and deny the key forever."""
    with pytest.raises(ValueError, match="refill_per_second"):
        TokenBucket(capacity=1, refill_per_second=-1.0)
