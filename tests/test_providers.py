from datetime import date

import httpx
import pytest

from adsignal.config import Settings
from adsignal.narrative.build import build_chain
from adsignal.narrative.prompt import build_prompt
from adsignal.narrative.providers.anthropic import AnthropicProvider
from adsignal.narrative.providers.gemini import GeminiProvider
from adsignal.narrative.providers.groq import GroqProvider
from adsignal.narrative.providers.jev import JevProvider
from adsignal.narrative.types import NarrativeRequest

SECRET = "sk-super-secret-key-do-not-leak"

REQUEST = NarrativeRequest(
    as_of=date(2024, 6, 1),
    campaign_count=12,
    anomaly_count=3,
    top_causes=("budget_pacing",),
    escalated_count=1,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _failing_client() -> httpx.Client:
    """A client whose transport raises a connection error carrying the URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connection refused: {request.url}", request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _timing_out_client() -> httpx.Client:
    """A client whose transport raises a timeout carrying the URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"timed out: {request.url}", request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_prompt_contains_the_facts_and_no_raw_rows() -> None:
    prompt = build_prompt(REQUEST)
    assert "2024-06-01" in prompt
    assert "12" in prompt and "3" in prompt
    assert "budget_pacing" in prompt


def test_gemini_returns_the_models_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "all quiet"}]}}]},
        )

    provider = GeminiProvider(api_key=SECRET, client=_client(handler))
    assert provider.generate(REQUEST) == "all quiet"


def test_gemini_error_never_contains_the_api_key() -> None:
    """Review Focus 1: httpx puts the request URL in its exception text.

    Gemini takes the key as a query parameter, so an unredacted error would
    carry the secret into logs and stored rows.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = GeminiProvider(api_key=SECRET, client=_client(handler))
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any type; content is the point
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "REDACTED" in message or "401" in message


def test_gemini_connection_error_never_contains_the_api_key() -> None:
    """The key is in the URL, so even a transport-level failure could leak it."""
    provider = GeminiProvider(api_key=SECRET, client=_failing_client())
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "connection" in message.lower()


def test_gemini_timeout_never_contains_the_api_key() -> None:
    """A timeout is a distinct failure category from a refused connection."""
    provider = GeminiProvider(api_key=SECRET, client=_timing_out_client())
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "timed out" in message.lower()


def test_gemini_raises_on_an_unexpected_payload_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = GeminiProvider(api_key=SECRET, client=_client(handler))
    with pytest.raises(ValueError, match="unexpected response"):
        provider.generate(REQUEST)


def test_groq_returns_the_models_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "all quiet"}}]},
        )

    provider = GroqProvider(api_key=SECRET, client=_client(handler))
    assert provider.generate(REQUEST) == "all quiet"


def test_groq_error_never_contains_the_api_key() -> None:
    """Groq sends the key as an `Authorization: Bearer` header, not a query param."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = GroqProvider(api_key=SECRET, client=_client(handler))
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "REDACTED" in message or "401" in message


def test_groq_connection_error_never_contains_the_api_key() -> None:
    provider = GroqProvider(api_key=SECRET, client=_failing_client())
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "connection" in message.lower()


def test_groq_timeout_never_contains_the_api_key() -> None:
    provider = GroqProvider(api_key=SECRET, client=_timing_out_client())
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "timed out" in message.lower()


def test_groq_raises_on_an_unexpected_payload_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = GroqProvider(api_key=SECRET, client=_client(handler))
    with pytest.raises(ValueError, match="unexpected response"):
        provider.generate(REQUEST)


def test_jev_returns_a_filled_template() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choice": "quiet"})

    provider = JevProvider(api_key=SECRET, client=_client(handler))
    text = provider.generate(REQUEST)
    assert isinstance(text, str) and text


def test_jev_error_never_contains_the_api_key() -> None:
    """Jev also sends the key as an `Authorization: Bearer` header."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = JevProvider(api_key=SECRET, client=_client(handler))
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "REDACTED" in message or "401" in message


def test_jev_connection_error_never_contains_the_api_key() -> None:
    provider = JevProvider(api_key=SECRET, client=_failing_client())
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "connection" in message.lower()


def test_jev_timeout_never_contains_the_api_key() -> None:
    provider = JevProvider(api_key=SECRET, client=_timing_out_client())
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "timed out" in message.lower()


def test_jev_raises_on_an_unexpected_payload_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = JevProvider(api_key=SECRET, client=_client(handler))
    with pytest.raises(ValueError, match="unexpected response"):
        provider.generate(REQUEST)


def test_anthropic_returns_the_models_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"content": [{"text": "all quiet"}]},
        )

    provider = AnthropicProvider(api_key=SECRET, client=_client(handler))
    assert provider.generate(REQUEST) == "all quiet"


def test_anthropic_error_never_contains_the_api_key() -> None:
    """Anthropic sends the key in an `x-api-key` header."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = AnthropicProvider(api_key=SECRET, client=_client(handler))
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "REDACTED" in message or "401" in message


def test_anthropic_connection_error_never_contains_the_api_key() -> None:
    provider = AnthropicProvider(api_key=SECRET, client=_failing_client())
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "connection" in message.lower()


def test_anthropic_timeout_never_contains_the_api_key() -> None:
    provider = AnthropicProvider(api_key=SECRET, client=_timing_out_client())
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        provider.generate(REQUEST)

    message = f"{excinfo.value}{excinfo.value.__cause__ or ''}"
    assert SECRET not in message, "the API key leaked into the error message"
    assert "timed out" in message.lower()


def test_anthropic_raises_on_an_unexpected_payload_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = AnthropicProvider(api_key=SECRET, client=_client(handler))
    with pytest.raises(ValueError, match="unexpected response"):
        provider.generate(REQUEST)


def test_build_chain_omits_providers_without_a_key() -> None:
    chain = build_chain(Settings(gemini_api_key="", groq_api_key=""))
    names = [provider.name for provider in chain._providers]  # noqa: SLF001
    assert "gemini" not in names
    assert names[-1] == "stub"


def test_build_chain_includes_a_configured_provider_first() -> None:
    chain = build_chain(Settings(gemini_api_key=SECRET))
    names = [provider.name for provider in chain._providers]  # noqa: SLF001
    assert names[0] == "gemini"
    assert names[-1] == "stub"


def test_build_chain_always_ends_with_the_stub() -> None:
    """The chain must never be empty, so a keyless deployment still renders."""
    chain = build_chain(Settings())
    result = chain.generate(REQUEST)
    assert result.provider == "stub"

    # Regression guard: `build_chain` once constructed the terminal provider
    # with a literal placeholder string, which permanently overrides the
    # provider's own summarizing default. A test that only checks `.provider`
    # would keep passing even if that literal came back — so assert on the
    # text itself, that it is a real summary of `REQUEST` rather than a stock
    # string.
    assert str(REQUEST.campaign_count) in result.text
    assert "placeholder" not in result.text.lower()
    assert "not configured" not in result.text.lower()


def test_build_chain_has_no_duplicate_provider_names() -> None:
    chain = build_chain(
        Settings(
            gemini_api_key=SECRET,
            groq_api_key=SECRET,
            jev_api_key=SECRET,
            anthropic_api_key=SECRET,
        )
    )
    names = [provider.name for provider in chain._providers]  # noqa: SLF001
    assert len(names) == len(set(names))
