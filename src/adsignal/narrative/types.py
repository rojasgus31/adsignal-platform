"""The contract every narrative provider implements."""

from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class NarrativeRequest:
    """The facts a narrative is written from.

    Deliberately a small, closed set of numbers rather than raw rows: the
    prompt is built from this, so nothing unexpected can reach a provider.
    """

    as_of: date
    campaign_count: int
    anomaly_count: int
    top_causes: tuple[str, ...]
    escalated_count: int


@dataclass(frozen=True, slots=True)
class NarrativeResult:
    """A generated narrative and the provenance a dashboard shows."""

    text: str
    provider: str
    latency_ms: float


@runtime_checkable
class NarrativeProvider(Protocol):
    """One backend capable of turning a request into prose."""

    name: str

    def generate(self, request: NarrativeRequest) -> str:
        """Return narrative text, or raise if this provider cannot."""
        ...
