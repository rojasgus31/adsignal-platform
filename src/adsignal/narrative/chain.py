"""An ordered chain of narrative providers with per-provider circuit breakers.

Providers are tried in order. A provider whose breaker is open is skipped
without being called, so one dead backend costs nothing on later requests. If
every provider fails, the caller gets one exception naming all of them — a
caller that cannot tell which backends it tried cannot diagnose anything.
"""

import time
from collections.abc import Sequence

from adsignal.narrative.breaker import CircuitBreaker
from adsignal.narrative.types import NarrativeProvider, NarrativeRequest, NarrativeResult


class AllProvidersFailed(RuntimeError):
    """Raised when no provider in the chain could produce a narrative."""


class NarrativeChain:
    """Tries each provider in turn until one succeeds."""

    def __init__(
        self,
        providers: Sequence[NarrativeProvider],
        failure_threshold: int = 2,
        cooldown_seconds: float = 60.0,
    ) -> None:
        if not providers:
            raise ValueError("a narrative chain needs at least one provider")
        self._providers = list(providers)
        self._breakers = {
            provider.name: CircuitBreaker(
                failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds
            )
            for provider in self._providers
        }

    def generate(self, request: NarrativeRequest) -> NarrativeResult:
        """Return the first successful narrative, or raise AllProvidersFailed."""
        attempted: list[str] = []

        for provider in self._providers:
            breaker = self._breakers[provider.name]
            if breaker.is_open():
                attempted.append(f"{provider.name} (breaker open)")
                continue

            started = time.monotonic()
            try:
                text = provider.generate(request)
            except Exception as error:  # noqa: BLE001 - any provider failure falls through
                breaker.record_failure()
                attempted.append(f"{provider.name} ({type(error).__name__})")
                continue

            breaker.record_success()
            return NarrativeResult(
                text=text,
                provider=provider.name,
                latency_ms=(time.monotonic() - started) * 1000.0,
            )

        raise AllProvidersFailed(f"every provider failed: {', '.join(attempted)}")
