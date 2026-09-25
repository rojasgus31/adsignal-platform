"""Anthropic's Messages API.

The API key travels in an `x-api-key` header alongside `anthropic-version`.
As with the other providers, header values can still reach exception text,
logs, and tracebacks, so every error path here is rewritten exactly as
Gemini's is, and each is raised with `from None` so the original exception is
not chained into the traceback.
"""

import httpx

from adsignal.narrative.prompt import build_prompt
from adsignal.narrative.types import NarrativeRequest

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    """Narrative generation via Claude Haiku."""

    def __init__(
        self,
        api_key: str,
        model: str = _MODEL,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "anthropic"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client

    def generate(self, request: NarrativeRequest) -> str:
        """Return generated text, raising with the key redacted on any failure."""
        payload = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": build_prompt(request)}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        client = self._client or httpx.Client(timeout=self._timeout)

        try:
            response = client.post(_ENDPOINT, headers=headers, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"anthropic returned {error.response.status_code} (headers REDACTED, key REDACTED)"
            ) from None
        except httpx.HTTPError:
            raise RuntimeError(
                "anthropic request failed (headers REDACTED, key REDACTED)"
            ) from None
        finally:
            if self._client is None:
                client.close()

        return _extract_text(response.json())


def _extract_text(payload: object) -> str:
    """Pull the generated text out, or say plainly that the shape was wrong."""
    try:
        content = payload["content"]  # type: ignore[index]
        return str(content[0]["text"])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"unexpected response shape from anthropic: {type(error).__name__}"
        ) from None
