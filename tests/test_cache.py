import pytest

from adsignal.api.cache import TtlCache


class FakeClock:
    """A clock the tests advance by hand, so expiry never needs a sleep."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_stored_value_comes_back() -> None:
    cache: TtlCache[str] = TtlCache(capacity=4, ttl_seconds=30.0, clock=FakeClock())
    cache.put("k", "v")
    assert cache.get("k") == "v"


def test_an_unknown_key_is_a_miss() -> None:
    cache: TtlCache[str] = TtlCache(capacity=4, ttl_seconds=30.0, clock=FakeClock())
    assert cache.get("absent") is None


def test_an_entry_expires_after_its_ttl() -> None:
    clock = FakeClock()
    cache: TtlCache[str] = TtlCache(capacity=4, ttl_seconds=30.0, clock=clock)
    cache.put("k", "v")

    clock.now = 29.0
    assert cache.get("k") == "v"

    clock.now = 30.0
    assert cache.get("k") is None


def test_the_cache_never_grows_past_its_capacity() -> None:
    """Finding 5: `/ask`'s key is caller-supplied free text, so it must be bounded.

    A TTL alone only causes a *miss* - it never evicts, so stale entries stay
    resident forever and a stream of distinct questions is a memory-exhaustion
    vector on the one publicly reachable live endpoint.
    """
    cache: TtlCache[int] = TtlCache(capacity=3, ttl_seconds=30.0, clock=FakeClock())
    for index in range(50):
        cache.put(f"question-{index}", index)

    assert len(cache) == 3
    assert cache.get("question-0") is None
    assert cache.get("question-49") == 49


def test_eviction_drops_the_least_recently_used_entry() -> None:
    cache: TtlCache[int] = TtlCache(capacity=2, ttl_seconds=30.0, clock=FakeClock())
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # "b" is now the least recently used
    cache.put("c", 3)

    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_a_capacity_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="capacity"):
        TtlCache(capacity=0, ttl_seconds=30.0)


def test_a_non_positive_ttl_is_rejected() -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        TtlCache(capacity=1, ttl_seconds=0.0)
