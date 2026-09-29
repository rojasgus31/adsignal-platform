"""Jev, a typed-value backend rather than a prose generator.

Jev does not return free text: it returns a `choice` selecting among a fixed
set of summary templates, which this provider then fills in with the facts
from the request. That is a narrower and less nuanced narrative than the
prose backends produce, which is exactly why Jev sits late in the chain —
behind Gemini and Groq, ahead of only the stub.

The API key travels as an `Authorization: Bearer` header. As with the other
providers, header values can still reach exception text, logs, and
tracebacks, so every error path here is rewritten exactly as Gemini's is,
and each is raised with `from None` so the original exception is not chained
into the traceback.
"""

import httpx

from adsignal.narrative.prompt import build_prompt
from adsignal.narrative.types import NarrativeRequest

_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

_TEMPLATES = {
    "quiet": "{as_of}: {campaign_count} campaigns running with no anomalies to report.",
    "attention": (
        "{as_of}: {anomaly_count} anomalies detected across {campaign_count} campaigns, "
        "most commonly {causes}."
    ),
    "escalation": (
        "{as_of}: {escalated_count} of {anomaly_count} anomalies were escalated for human "
        "review, most commonly {causes}."
    ),
}


class JevProvider:
    """Narrative generation via Jev's typed-choice API."""

    def __init__(
        self,
        api_key: str,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "jev"
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    def generate(self, request: NarrativeRequest) -> str:
        """Return a filled template, raising with the key redacted on any failure."""
        payload = {
            "prompt": build_prompt(request),
            "choices": list(_TEMPLATES),
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        client = self._client or httpx.Client(timeout=self._timeout)

        # The failure is recorded here and raised *below*, outside every
        # `except` block. `raise ... from None` clears `__cause__` but leaves
        # `__context__` pointing at the httpx exception, so the chain still
        # reaches whatever that exception's text contains. This provider sends
        # its key in a header rather than in the URL, so no credential is
        # retained either way - but this is the pattern to copy, and the next
        # provider added here will copy whatever is written.
        failure: str | None = None
        response: httpx.Response | None = None

        try:
            try:
                response = client.post(
                    _ENDPOINT, headers=headers, json=payload, timeout=self._timeout
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                failure = (
                    f"jev returned {error.response.status_code} (headers REDACTED, key REDACTED)"
                )
            except httpx.TimeoutException:
                failure = "jev request timed out (headers REDACTED, key REDACTED)"
            except httpx.ConnectError:
                failure = "jev connection failed (headers REDACTED, key REDACTED)"
            except httpx.HTTPError:
                failure = "jev request failed (headers REDACTED, key REDACTED)"
        finally:
            if self._client is None:
                client.close()

        if failure is not None:
            raise RuntimeError(failure)
        if response is None:  # pragma: no cover - unreachable while failure is None
            raise RuntimeError("jev returned no response (key REDACTED)")

        return _fill_template(response.json(), request)


def _fill_template(payload: object, request: NarrativeRequest) -> str:
    """Look up the chosen template and fill it from the request, or say the shape was wrong."""
    try:
        choice = payload["choice"]  # type: ignore[index]
        template = _TEMPLATES[choice]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"unexpected response shape from jev: {type(error).__name__}") from None

    causes = ", ".join(request.top_causes) if request.top_causes else "none"
    return template.format(
        as_of=request.as_of.isoformat(),
        campaign_count=request.campaign_count,
        anomaly_count=request.anomaly_count,
        causes=causes,
        escalated_count=request.escalated_count,
    )
