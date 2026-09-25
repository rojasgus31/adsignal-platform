"""A deterministic provider for tests.

Every test in this project drives the chain through this rather than a live
API: the fallback logic is what needs proving, and proving it must not depend
on a network or a key.
"""

from adsignal.narrative.types import NarrativeRequest


class StubProvider:
    """Returns fixed text, or raises a fixed exception, and counts its calls."""

    def __init__(
        self,
        text: str = "stub narrative",
        fail_with: Exception | None = None,
        name: str = "stub",
    ) -> None:
        self.name = name
        self._text = text
        self._fail_with = fail_with
        self.call_count = 0

    def generate(self, request: NarrativeRequest) -> str:
        """Return the configured text, or raise the configured error."""
        self.call_count += 1
        if self._fail_with is not None:
            raise self._fail_with
        return self._text
