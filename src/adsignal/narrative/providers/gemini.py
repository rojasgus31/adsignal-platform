"""Google Gemini via the Generative Language REST API.

The API key travels as a query parameter, which means httpx would put it in
the URL of any exception it raises — including transport-level failures such
as a timeout or a refused connection, not only HTTP status errors. Every
error path here is rewritten so the secret cannot reach a log, a stored row,
or a response, and each is raised with `from None` so the original exception
(which may carry the URL) is not chained into the traceback.
"""

import httpx

from adsignal.narrative.prompt import build_prompt
from adsignal.narrative.types import NarrativeRequest

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiProvider:
    """Narrative generation via Gemini Flash."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
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
        url = _ENDPOINT.format(model=self._model)
        payload = {"contents": [{"parts": [{"text": build_prompt(request)}]}]}
        client = self._client or httpx.Client(timeout=self._timeout)

        try:
            response = client.post(
                url, params={"key": self._api_key}, json=payload, timeout=self._timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"gemini returned {error.response.status_code} (url REDACTED, key REDACTED)"
            ) from None
        except httpx.TimeoutException:
            raise RuntimeError("gemini request timed out (url REDACTED, key REDACTED)") from None
        except httpx.ConnectError:
            raise RuntimeError("gemini connection failed (url REDACTED, key REDACTED)") from None
        except httpx.HTTPError:
            raise RuntimeError("gemini request failed (url REDACTED, key REDACTED)") from None
        finally:
            if self._client is None:
                client.close()

        return _extract_text(response.json())


def _extract_text(payload: object) -> str:
    """Pull the generated text out, or say plainly that the shape was wrong."""
    try:
        candidates = payload["candidates"]  # type: ignore[index]
        return str(candidates[0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"unexpected response shape from gemini: {type(error).__name__}") from None
