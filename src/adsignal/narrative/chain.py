"""An ordered chain of narrative providers with per-provider circuit breakers.

Providers are tried in order. A provider whose breaker is open is skipped
without being called, so one dead backend costs nothing on later requests. If
every provider fails, the caller gets one exception naming all of them — a
caller that cannot tell which backends it tried cannot diagnose anything.

The chain also imposes a **total budget** across every provider, not just the
per-provider timeout each one carries. A per-provider timeout does not bound
the chain: httpx applies its value per phase (connect/read/write/pool), and
four configured providers each allowed 20 s can stall the caller for 80 s or
more before the stub is reached. A provider that answers only after the
budget has run out is recorded as a *failure*, not a success, so its breaker
opens and later runs skip it — a backend slow enough to blow the whole
budget is as unusable as one that raises, and nothing else would ever open
its breaker.
"""

import time
from collections.abc import Callable, Sequence

from adsignal.narrative.breaker import CircuitBreaker
from adsignal.narrative.types import NarrativeProvider, NarrativeRequest, NarrativeResult

# Generous enough that a healthy first provider is never cut off, short
# enough that a dashboard's nightly job is not held for minutes by a chain of
# stalled backends. `build_chain` overrides it from settings.
DEFAULT_TOTAL_BUDGET_SECONDS = 45.0


class AllProvidersFailed(RuntimeError):
    """Raised when no provider in the chain could produce a narrative."""


class NarrativeChain:
    """Tries each provider in turn until one succeeds."""

    def __init__(
        self,
        providers: Sequence[NarrativeProvider],
        failure_threshold: int = 2,
        cooldown_seconds: float = 60.0,
        total_budget_seconds: float = DEFAULT_TOTAL_BUDGET_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("a narrative chain needs at least one provider")
        if total_budget_seconds <= 0:
            raise ValueError("total_budget_seconds must be positive")
        self._total_budget_seconds = total_budget_seconds
        self._clock = clock
        self._providers = list(providers)
        self._raise_if_duplicate_names(self._providers)
        self._breakers = {
            provider.name: CircuitBreaker(
                failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds
            )
            for provider in self._providers
        }

    def generate(self, request: NarrativeRequest) -> NarrativeResult:
        """Return the first successful narrative, or raise AllProvidersFailed."""
        deadline = self._clock() + self._total_budget_seconds
        attempted: list[str] = []

        for provider in self._providers:
            result, reason = self._attempt(provider, request, deadline)
            if result is not None:
                return result
            attempted.append(reason)

        raise AllProvidersFailed(f"every provider failed: {', '.join(attempted)}")

    def _attempt(
        self, provider: NarrativeProvider, request: NarrativeRequest, deadline: float
    ) -> tuple[NarrativeResult | None, str]:
        """Try one provider, returning its result or why it did not produce one."""
        breaker = self._breakers[provider.name]
        if breaker.is_open():
            return None, f"{provider.name} (breaker open)"
        if self._clock() >= deadline:
            return None, f"{provider.name} (chain budget exhausted)"

        started = self._clock()
        try:
            text = provider.generate(request)
        except Exception as error:  # noqa: BLE001 - any provider failure falls through
            breaker.record_failure()
            return None, f"{provider.name} ({type(error).__name__})"

        finished = self._clock()
        if finished > deadline:
            # A late answer is a failed one: recording it as a success would
            # leave the breaker of a permanently slow backend closed forever.
            breaker.record_failure()
            return None, f"{provider.name} (exceeded the chain budget)"

        breaker.record_success()
        result = NarrativeResult(
            text=text, provider=provider.name, latency_ms=(finished - started) * 1000.0
        )
        return result, ""

    @staticmethod
    def _raise_if_duplicate_names(providers: list[NarrativeProvider]) -> None:
        """Reject providers sharing a name.

        The name is the breaker map's key and the provenance recorded on a
        result, so two providers sharing one would silently share a circuit
        breaker: a failure against one would open the breaker gating the
        other, and a success for either would reset it for both.
        """
        seen: set[str] = set()
        duplicates: set[str] = set()
        for provider in providers:
            if provider.name in seen:
                duplicates.add(provider.name)
            seen.add(provider.name)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"provider names must be unique, duplicated: {names}")
