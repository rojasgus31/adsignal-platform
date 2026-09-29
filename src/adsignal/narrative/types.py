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


class ProviderUnavailable(RuntimeError):
    """A provider failure whose message was written to be shown to a user.

    `narrative.providers.gemini.generate_text` composes every one of its
    failure strings itself — a status code or a transport verb, with the URL
    and the key replaced by the literal word REDACTED — precisely so they can
    be surfaced. This type is how a caller tells those strings apart from an
    arbitrary exception's text, which is never safe: an httpx error carries
    the request URL, and the key travels in that URL.

    `narrative.sqlgen.SqlGenerator` surfaces the message of this type and the
    *type name* of everything else, so raising it is a claim that the message
    was redacted by construction. Nothing else in the project raises it.

    A `RuntimeError` subclass, so every existing `except RuntimeError` and
    every catch-all still behaves exactly as it did.

    `status` is the HTTP status when the provider answered with one, and
    `None` for a transport failure that never got a response (a timeout, a
    refused connection). It exists so a caller can tell *which* failure this
    is without parsing the message: `narrative.sqlgen` retries a 503 and
    nothing else, and a regex over prose is not a thing to hang that on.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status
