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
    skipped: tuple[str, ...] = ()
    """Providers tried before this one, and why each did not answer.

    Without this, "stub answered" is the only thing a reader ever sees, and
    it looks identical whether no key was configured or a real provider
    returned 404 on every call — which is exactly what a retired model
    looks like. Each entry is the string `NarrativeChain` already composes:
    a provider name plus an exception *type* or a chain reason, never an
    exception's message, so it carries no key and no URL.

    Defaulted so every existing construction of this dataclass — including
    the stored-narrative read path, which has no attempts to report — stays
    valid.
    """


@runtime_checkable
class NarrativeProvider(Protocol):
    """One backend capable of turning a request into prose."""

    name: str

    def generate(self, request: NarrativeRequest) -> str:
        """Return narrative text, or raise if this provider cannot."""
        ...
