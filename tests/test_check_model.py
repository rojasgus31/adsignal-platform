"""The check that would have caught a shipped model name that does not work.

Two defaults have now been wrong: `gemini-2.0-flash` was retired (404 to a
valid key), and `gemini-3.5-flash` answered 503 to every call while three of
its siblings answered fine. Both were chosen from a live `ListModels`
response and both passed the whole suite, because every test that touched a
model name used a mocked transport.

These tests are mocked too — they have to be; a suite that calls Google is a
suite that costs money and fails when Google does. What they pin is the
*command*: that it reports a real answer as an answer, reports a 503 as a
503 rather than as a type name, and never prints the key. The command itself
is the unmocked check, and it is documented beside the deployment steps.
"""

import httpx
import pytest

from adsignal.cli.check_model import ModelCheck, check_model, main
from adsignal.config import DEFAULT_GEMINI_MODEL, Settings, get_settings

SECRET = "sk-check-model-key-do-not-leak"
MODEL = "gemini-3.6-flash"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _answering(text: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})

    return _client(handler)


def _status(code: int) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"error": {"message": "nope"}})

    return _client(handler)


def test_a_model_that_answers_is_reported_as_answering() -> None:
    result = check_model(SECRET, MODEL, client=_answering("ok"))
    assert result == ModelCheck(model=MODEL, answered=True, detail="answered with 2 characters")


def test_a_503_is_reported_as_a_503_not_as_an_exception_type() -> None:
    """The whole reason this command exists: "RuntimeError" diagnosed nothing."""
    result = check_model(SECRET, MODEL, client=_status(503))
    assert result.answered is False
    assert "503" in result.detail


def test_a_retired_model_reads_differently_from_an_outage() -> None:
    outage = check_model(SECRET, MODEL, client=_status(503))
    retired = check_model(SECRET, MODEL, client=_status(404))
    assert outage.detail != retired.detail
    assert "404" in retired.detail


def test_a_response_with_no_generated_text_is_not_an_answer() -> None:
    """An accepted request that generates nothing is still a broken model."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    result = check_model(SECRET, MODEL, client=_client(handler))
    assert result.answered is False
    assert "no generated text" in result.detail


@pytest.mark.parametrize("code", [400, 401, 403, 404, 429, 500, 503])
def test_no_failure_detail_ever_carries_the_key_or_the_url(code: int) -> None:
    result = check_model(SECRET, MODEL, client=_status(code))
    assert SECRET not in result.detail
    assert "key=" not in result.detail
    assert "generativelanguage" not in result.detail
    assert "https://" not in result.detail


def test_the_check_calls_the_configured_model_and_nothing_else() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    check_model(SECRET, "gemini-3.8-flash", client=_client(handler))
    assert "gemini-3.8-flash" in seen["path"]


def test_no_key_configured_is_an_exit_code_not_a_false_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A check that passes without calling anything is the bug, not the fix."""
    monkeypatch.setenv("ADSIGNAL_GEMINI_API_KEY", "")
    get_settings.cache_clear()
    try:
        assert main() == 2
    finally:
        get_settings.cache_clear()


def test_the_shipped_default_is_the_one_the_sweep_found_answering() -> None:
    """`gemini-3.5-flash` and `gemini-3.7-flash` returned 503 on every attempt."""
    assert DEFAULT_GEMINI_MODEL not in ("gemini-2.0-flash", "gemini-3.5-flash", "gemini-3.7-flash")
    assert Settings(_env_file=None).gemini_model == DEFAULT_GEMINI_MODEL
