"""A minimal circuit breaker.

Pure state: no I/O, no threads, and the clock is injected so tests do not
sleep. Opening on repeated failure is the easy half; the half that matters is
closing again — a breaker with no recovery path converts a thirty-second
provider blip into permanent degradation, and no happy-path test notices.
"""

import time
from collections.abc import Callable


class CircuitBreaker:
    """Opens after consecutive failures, closes again after a cooldown."""

    def __init__(
        self,
        failure_threshold: int = 2,
        cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")

        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def consecutive_failures(self) -> int:
        """How many failures have happened since the last success."""
        return self._consecutive_failures

    def is_open(self) -> bool:
        """True while this provider should be skipped.

        Checking also performs recovery: once the cooldown has elapsed the
        breaker closes, so the next call gives the provider another chance.
        """
        if self._opened_at is None:
            return False
        if self._clock() - self._opened_at >= self._cooldown_seconds:
            self._reset()
            return False
        return True

    def record_success(self) -> None:
        """Clear the failure count and close the breaker."""
        self._reset()

    def record_failure(self) -> None:
        """Count a failure, opening the breaker at the threshold.

        The cooldown is timestamped only by the failure that crosses the
        threshold. A failure recorded while the breaker is already open
        leaves `_opened_at` untouched, so the cooldown still runs from the
        original opening — it is not extended by further failures.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold and self._opened_at is None:
            self._opened_at = self._clock()

    def _reset(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None
