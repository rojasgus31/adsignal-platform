"""Question -> SQL: the prompt, the provider, and the no-key state.

`ADSIGNAL_GEMINI_API_KEY` is unset in this repository's own environment, so
"no model is configured" is the path that actually runs today. It is tested
first and hardest here: it must be a state the endpoint can describe, never
an exception and never a fabricated answer.
"""

import httpx
import pytest

from adsignal.config import DEFAULT_GEMINI_MODEL, Settings
from adsignal.narrative.providers.gemini_sql import GeminiSqlProvider
from adsignal.narrative.sqlgen import (
    SqlGenerationFailed,
    SqlGenerator,
    SqlQuestion,
    build_sql_generator,
    build_sql_prompt,
    extract_sql,
)

SECRET = "sk-super-secret-key-do-not-leak"
MODEL = DEFAULT_GEMINI_MODEL

QUESTION = SqlQuestion(
    question="which channel spent the most last month?",
    schema_description="fct_campaign_daily(channel TEXT, spend DOUBLE PRECISION)",
)


def _chain(error: BaseException) -> str:
    """Every message reachable from `error`, following both exception links.

    `raise ... from None` suppresses `__cause__` but leaves `__context__`
    pointing at the exception being handled — here, the httpx error whose
    text contains the request URL, and the key travels in that URL as a query
    parameter. A leak test that reads only `__cause__` cannot fail.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{current!r}")
        current = current.__cause__ or current.__context__
    return " ".join(parts)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _answering(sql: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": sql}]}}]})

    return _client(handler)


class _StubProvider:
    """A provider that returns a fixed string, or raises a fixed error."""

    def __init__(self, sql: str = "SELECT 1", error: Exception | None = None) -> None:
        self.name = "stub"
        self._sql = sql
        self._error = error
        self.calls = 0

    def generate_sql(self, request: SqlQuestion) -> str:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._sql


def test_no_key_means_no_generator_rather_than_an_error() -> None:
    """The state this deployment is in today. It must be representable."""
    assert build_sql_generator(Settings(gemini_api_key="")) is None


def test_a_configured_key_produces_a_gemini_generator() -> None:
    generator = build_sql_generator(Settings(gemini_api_key=SECRET))
    assert generator is not None
    assert generator.name == "gemini"


def test_building_a_generator_never_echoes_the_key() -> None:
    """The key reaches the provider and stops there; repr must not carry it."""
    generator = build_sql_generator(Settings(gemini_api_key=SECRET))
    assert generator is not None
    assert SECRET not in repr(generator)
    assert SECRET not in str(generator)


def test_the_prompt_carries_the_question_and_the_schema_and_nothing_else() -> None:
    prompt = build_sql_prompt(QUESTION)
    assert QUESTION.question in prompt
    assert "fct_campaign_daily" in prompt
    assert "postgresql://" not in prompt
    assert "password" not in prompt.lower()
    assert SECRET not in prompt


def test_the_prompt_states_the_rules_the_validator_enforces() -> None:
    """The prompt is a quality measure: it should ask for what will be accepted."""
    prompt = build_sql_prompt(QUESTION).lower()
    assert "select" in prompt
    assert "limit" in prompt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SELECT 1", "SELECT 1"),
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("  SELECT 1  \n", "SELECT 1"),
        ("```SQL\nSELECT 1\n```", "SELECT 1"),
    ],
)
def test_a_fenced_answer_is_unwrapped(raw: str, expected: str) -> None:
    """Models fence their code. The fence is not SQL and must not reach the database."""
    assert extract_sql(raw) == expected


def test_the_generator_returns_the_models_sql_and_its_name() -> None:
    generator = SqlGenerator(_StubProvider(sql="SELECT channel FROM fct_campaign_daily"))
    generated = generator.generate(QUESTION)
    assert generated.sql == "SELECT channel FROM fct_campaign_daily"
    assert generated.provider == "stub"


def test_a_provider_failure_becomes_a_typed_failure() -> None:
    generator = SqlGenerator(_StubProvider(error=RuntimeError("gemini returned 503")))
    with pytest.raises(SqlGenerationFailed):
        generator.generate(QUESTION)


def test_a_failing_provider_is_skipped_once_its_breaker_opens() -> None:
    """Reused from `narrative.breaker`: a dead backend stops costing a round trip."""
    provider = _StubProvider(error=RuntimeError("down"))
    generator = SqlGenerator(provider, failure_threshold=2, cooldown_seconds=60.0)
    for _ in range(4):
        with pytest.raises(SqlGenerationFailed):
            generator.generate(QUESTION)
    assert provider.calls == 2, "the breaker never opened"


def test_a_generation_failure_never_carries_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    generator = SqlGenerator(
        GeminiSqlProvider(api_key=SECRET, model=MODEL, client=_client(handler))
    )
    with pytest.raises(SqlGenerationFailed) as failed:
        generator.generate(QUESTION)
    # `from None` clears `__cause__` but NOT `__context__`, which still holds
    # the httpx exception whose URL carries the key. Both have to be clean.
    message = _chain(failed.value)
    assert SECRET not in message
    assert "key=" not in message


def test_the_gemini_provider_returns_unwrapped_sql() -> None:
    provider = GeminiSqlProvider(
        api_key=SECRET, model=MODEL, client=_answering("```sql\nSELECT 1 FROM dim_campaign\n```")
    )
    assert provider.generate_sql(QUESTION) == "SELECT 1 FROM dim_campaign"


def test_the_gemini_provider_sends_the_prompt_and_not_the_schema_owner() -> None:
    """What leaves the process is the question and the table shapes. Nothing else."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "x"}]}}]})

    GeminiSqlProvider(api_key=SECRET, model=MODEL, client=_client(handler)).generate_sql(QUESTION)
    body = str(seen["body"])
    assert "fct_campaign_daily" in body
    assert SECRET not in body
    assert "postgresql://" not in body


def test_a_timeout_is_reported_without_the_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"timed out: {request.url}", request=request)

    provider = GeminiSqlProvider(api_key=SECRET, model=MODEL, client=_client(handler))
    with pytest.raises(RuntimeError) as failed:
        provider.generate_sql(QUESTION)
    message = _chain(failed.value)
    assert SECRET not in message
    assert "generativelanguage" not in message


def test_the_model_is_configuration_not_a_literal() -> None:
    """A retired model must be a one-line env change, not a code edit.

    `gemini-2.0-flash` was hardcoded in both providers until it was retired,
    at which point a valid key returned 404 on every call. Neither provider
    carries a default now: the name comes from `Settings.gemini_model`.
    """
    chosen = "gemini-3.8-flash"
    generator = build_sql_generator(Settings(gemini_api_key=SECRET, gemini_model=chosen))
    assert generator is not None

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "x"}]}}]})

    GeminiSqlProvider(api_key=SECRET, model=chosen, client=_client(handler)).generate_sql(QUESTION)
    assert chosen in seen["path"]


def test_the_shipped_model_default_is_current() -> None:
    """The retired name must not come back by copy-paste."""
    assert DEFAULT_GEMINI_MODEL != "gemini-2.0-flash"
    assert Settings().gemini_model == DEFAULT_GEMINI_MODEL
