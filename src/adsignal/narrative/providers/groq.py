"""Groq's OpenAI-compatible chat completions API.

The API key travels as an `Authorization: Bearer` header rather than a query
parameter, but a header value can still reach exception text, logs, and
tracebacks depending on how errors are constructed — so every error path
here is rewritten exactly as Gemini's is, and each is raised with `from None`
so the original exception is not chained into the traceback.
"""

import httpx

from adsignal.narrative.prompt import build_prompt
from adsignal.narrative.types import NarrativeRequest

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "openai/gpt-oss-120b"


class GroqProvider:
    """Narrative generation via Groq."""

    def __init__(
        self,
        api_key: str,
        model: str = _MODEL,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "groq"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client

    def generate(self, request: NarrativeRequest) -> str:
        """Return generated text, raising with the key redacted on any failure."""
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": build_prompt(request)}],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        client = self._client or httpx.Client(timeout=self._timeout)

        try:
            response = client.post(_ENDPOINT, headers=headers, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"groq returned {error.response.status_code} (headers REDACTED, key REDACTED)"
            ) from None
        except httpx.TimeoutException:
            raise RuntimeError("groq request timed out (headers REDACTED, key REDACTED)") from None
        except httpx.ConnectError:
            raise RuntimeError("groq connection failed (headers REDACTED, key REDACTED)") from None
        except httpx.HTTPError:
            raise RuntimeError("groq request failed (headers REDACTED, key REDACTED)") from None
        finally:
            if self._client is None:
                client.close()

        return _extract_text(response.json())


def _extract_text(payload: object) -> str:
    """Pull the generated text out, or say plainly that the shape was wrong."""
    try:
        choices = payload["choices"]  # type: ignore[index]
        return str(choices[0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"unexpected response shape from groq: {type(error).__name__}") from None
