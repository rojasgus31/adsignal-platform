"""Google Gemini via the Generative Language REST API.

Two callers share this module: `GeminiProvider`, which writes the daily
narrative, and `narrative.providers.gemini_sql.GeminiSqlProvider`, which
turns an `/ask` question into SQL. They send different prompts to the same
endpoint, so the transport and — more importantly — the redaction below live
in `generate_text` and are written once.

Neither class defaults `model`: the name belongs to whoever configures the
deployment (`Settings.gemini_model`, `ADSIGNAL_GEMINI_MODEL`), because a
default baked in here is a claim about Google's catalogue that goes stale
silently — `gemini-2.0-flash` was this project's default until it was
retired and started answering 404 to a valid key.

The API key travels as a query parameter, which means httpx would put it in
the URL of any exception it raises — including transport-level failures such
as a timeout or a refused connection, not only HTTP status errors. Every
error path here is rewritten so the secret cannot reach a log, a stored row,
or a response, and each is raised with `from None` so the original exception
(which may carry the URL) is not chained into the traceback.
"""

import httpx

from adsignal.narrative.prompt import build_prompt
from adsignal.narrative.types import (
    NarrativeRequest,
    ProviderFailureCause,
    ProviderUnavailable,
)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def generate_text(
    prompt: str,
    api_key: str,
    model: str,
    timeout: float,
    client: httpx.Client | None = None,
) -> str:
    """POST `prompt` to Gemini and return the text, with the key redacted on failure.

    Every exception raised here is a fresh `ProviderUnavailable` raised
    outside the `except` block that recorded it, so neither `__cause__` nor
    `__context__` links back to the httpx exception. httpx puts the request
    URL in its own exception text and the key travels in that URL as a query
    parameter, so an exception chain that reaches it is an exception chain
    that leaks it.

    The type, rather than a bare `RuntimeError`, is what lets a caller show
    the message: the four strings below name a status code or a transport
    failure and nothing else, and `narrative.types.ProviderUnavailable` says
    so. Before it, `SqlGenerator` could only report `RuntimeError`, so a
    transient 503 and an expired key reached the user identically.
    """
    url = _ENDPOINT.format(model=model)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    http = client or httpx.Client(timeout=timeout)

    # The failure is recorded here and raised *below*, outside every `except`
    # block. `raise ... from None` clears `__cause__` but leaves
    # `__context__` pointing at the httpx exception whose text contains the
    # request URL - and the key travels in that URL as a query parameter.
    # Standard traceback printing honours `__suppress_context__` and hides
    # it, but anything that walks the chain itself (a structured logger, an
    # error reporter, a test) can still read the key out of it. Raising with
    # no exception being handled is what makes `__context__` genuinely None.
    failure: str | None = None
    # The status carried on the raised `ProviderUnavailable`. Set only where
    # the server actually answered; a timeout or a refused connection has no
    # status, and `None` is what stops a caller retrying those.
    status: int | None = None
    # Which kind of failure this was, recorded where it is known rather than
    # recovered later by matching on the message. `narrative.sqlgen` writes one
    # user-facing sentence per value.
    cause: ProviderFailureCause = "unspecified"
    response: httpx.Response | None = None

    try:
        try:
            response = http.post(url, params={"key": api_key}, json=payload, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            cause = "status"
            failure = f"gemini returned {status} (url REDACTED, key REDACTED)"
        except httpx.TimeoutException:
            cause = "timeout"
            failure = "gemini request timed out (url REDACTED, key REDACTED)"
        except httpx.ConnectError:
            cause = "unreachable"
            failure = "gemini connection failed (url REDACTED, key REDACTED)"
        except httpx.HTTPError:
            cause = "transport"
            failure = "gemini request failed (url REDACTED, key REDACTED)"
    finally:
        if client is None:
            http.close()

    if failure is not None:
        raise ProviderUnavailable(failure, status, cause)
    if response is None:  # pragma: no cover - unreachable while failure is None
        raise ProviderUnavailable("gemini returned no response (url REDACTED, key REDACTED)")

    return _extract_text(response.json())


class GeminiProvider:
    """Narrative generation via Gemini Flash."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "gemini"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client

    def generate(self, request: NarrativeRequest) -> str:
        """Return generated text, raising with the key redacted on any failure."""
        return generate_text(
            build_prompt(request),
            api_key=self._api_key,
            model=self._model,
            timeout=self._timeout,
            client=self._client,
        )


def _extract_text(payload: object) -> str:
    """Pull the generated text out, or say plainly that the shape was wrong."""
    try:
        candidates = payload["candidates"]  # type: ignore[index]
        return str(candidates[0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"unexpected response shape from gemini: {type(error).__name__}") from None
