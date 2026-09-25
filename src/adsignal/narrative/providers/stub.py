"""The deterministic, local provider — the chain's terminal fallback.

Every test in this project can also drive the chain through this with an
explicit literal `text`, rather than a live API: the fallback logic is what
needs proving, and proving it must not depend on a network or a key.

Left at its default, it does not return a stock literal. It writes a real,
deterministic summary of the request via `summarize()`, so a deployment with
no provider configured renders a readable narrative instead of placeholder
text.
"""

from adsignal.narrative.summary import summarize
from adsignal.narrative.types import NarrativeRequest

_USE_SUMMARY = object()


class StubProvider:
    """Returns fixed text, a computed summary, or a fixed exception.

    Counts its calls either way.
    """

    def __init__(
        self,
        text: object = _USE_SUMMARY,
        fail_with: Exception | None = None,
        name: str = "stub",
    ) -> None:
        self.name = name
        self._text = text
        self._fail_with = fail_with
        self.call_count = 0

    def generate(self, request: NarrativeRequest) -> str:
        """Return the configured text, a computed summary, or raise."""
        self.call_count += 1
        if self._fail_with is not None:
            raise self._fail_with
        if self._text is _USE_SUMMARY:
            return summarize(request)
        assert isinstance(self._text, str)
        return self._text
