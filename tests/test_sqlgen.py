"""Question -> SQL: the prompt, the provider, and the no-key state.

`ADSIGNAL_GEMINI_API_KEY` is unset in this repository's own environment, so
"no model is configured" is the path that actually runs today. It is tested
first and hardest here: it must be a state the endpoint can describe, never
an exception and never a fabricated answer.
"""

from datetime import date

import httpx
import pytest

from adsignal.config import DEFAULT_GEMINI_MODEL, Settings
from adsignal.narrative.providers.gemini import GeminiProvider
from adsignal.narrative.providers.gemini_sql import GeminiSqlProvider
from adsignal.narrative.sqlgen import (
    SqlGenerationFailed,
    SqlGenerator,
    SqlQuestion,
    build_sql_generator,
    build_sql_prompt,
    extract_sql,
)
from adsignal.narrative.types import NarrativeRequest, ProviderUnavailable

SECRET = "sk-super-secret-key-do-not-leak"

# The narrative contract's own request, for the one test that proves the
# `/ask` retry did not leak into the narrative path.
NARRATIVE_REQUEST = NarrativeRequest(
    as_of=date(2025, 12, 31),
    campaign_count=1,
    anomaly_count=0,
    top_causes=(),
    escalated_count=0,
)
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


# --- what a refusal is allowed to say about why ----------------------------
#
# The defect: `SqlGenerator` reported the exception's *type name*, so a
# transient 503, a rate limit, an expired key and a retired model all reached
# the user as "gemini could not answer (RuntimeError)". `generate_text`
# composes those messages with the URL and the key already replaced by the
# word REDACTED, and `ProviderUnavailable` is how it says so.


def test_a_redacted_provider_message_reaches_the_user_whole() -> None:
    error = ProviderUnavailable("gemini returned 503 (url REDACTED, key REDACTED)")
    generator = SqlGenerator(_StubProvider(error=error))
    with pytest.raises(SqlGenerationFailed) as failed:
        generator.generate(QUESTION)
    assert str(failed.value) == "gemini returned 503 (url REDACTED, key REDACTED)"


def test_the_status_code_is_what_distinguishes_one_outage_from_another() -> None:
    """503 and 401 are different problems and must not read identically."""
    transient = SqlGenerator(_StubProvider(error=ProviderUnavailable("gemini returned 503 (x)")))
    expired = SqlGenerator(_StubProvider(error=ProviderUnavailable("gemini returned 401 (x)")))
    with pytest.raises(SqlGenerationFailed) as first:
        transient.generate(QUESTION)
    with pytest.raises(SqlGenerationFailed) as second:
        expired.generate(QUESTION)
    assert str(first.value) != str(second.value)


def test_an_ordinary_exceptions_text_is_never_passed_through() -> None:
    """The constraint the fix must not break: only messages proven redacted.

    A `psycopg` error, a driver message or a stray `RuntimeError` can carry a
    DSN. None of them is a `ProviderUnavailable`, so none of them contributes
    anything but a type name.
    """
    dsn = f"postgresql://adsignal:{SECRET}@db.internal:5432/adsignal"
    generator = SqlGenerator(_StubProvider(error=RuntimeError(f"connection to {dsn} failed")))
    with pytest.raises(SqlGenerationFailed) as failed:
        generator.generate(QUESTION)
    message = _chain(failed.value)
    assert SECRET not in message
    assert "postgresql://" not in message
    assert "db.internal" not in message
    assert str(failed.value) == "stub could not answer (RuntimeError)"


def test_a_real_gemini_outage_names_its_status_and_still_hides_the_key() -> None:
    """End to end through the transport: the 503 survives, the URL and key do not."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "backend overloaded"}})

    generator = SqlGenerator(
        GeminiSqlProvider(api_key=SECRET, model=MODEL, client=_client(handler))
    )
    with pytest.raises(SqlGenerationFailed) as failed:
        generator.generate(QUESTION)
    assert "503" in str(failed.value)
    message = _chain(failed.value)
    assert SECRET not in message
    assert "key=" not in message
    assert "generativelanguage" not in message
    assert "https://" not in message


def test_a_provider_error_the_model_layer_did_not_write_is_still_typed_only() -> None:
    """A malformed Gemini body raises `ValueError`, which is not passed through."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nothing": "useful"})

    generator = SqlGenerator(
        GeminiSqlProvider(api_key=SECRET, model=MODEL, client=_client(handler))
    )
    with pytest.raises(SqlGenerationFailed) as failed:
        generator.generate(QUESTION)
    assert str(failed.value) == "gemini could not answer (ValueError)"


# --- the single retry on an intermittent 503 -------------------------------
#
# Gemini's free tier returns 503 at roughly one call in twelve, measured six
# calls apart through `GeminiSqlProvider`. The owner met three in a row and
# concluded the feature was broken. One retry takes a user-visible failure
# from ~8% to under 1%; the cost is bounded because a 503 arrives fast and
# nothing else is retried.


class _SequenceProvider:
    """A `SqlProvider` that replays a scripted list of outcomes, then repeats.

    Each entry is either SQL to return or an exception to raise, so a test
    can say "503 then success" and "503 then 503" exactly.
    """

    def __init__(self, *outcomes: str | Exception) -> None:
        self.name = "stub"
        self._outcomes = list(outcomes)
        self.calls = 0

    def generate_sql(self, request: SqlQuestion) -> str:
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _unavailable(status: int | None) -> ProviderUnavailable:
    return ProviderUnavailable(f"gemini returned {status} (url REDACTED, key REDACTED)", status)


def _generator(provider: _SequenceProvider, **kwargs: object) -> SqlGenerator:
    """A generator whose backoff costs the suite nothing."""
    slept: list[float] = []
    generator = SqlGenerator(provider, sleep=slept.append, **kwargs)  # type: ignore[arg-type]
    generator.slept = slept  # type: ignore[attr-defined]
    return generator


def test_a_503_is_tried_once_more_and_the_second_answer_is_used() -> None:
    provider = _SequenceProvider(_unavailable(503), "SELECT 1 FROM dim_campaign")
    generated = _generator(provider).generate(QUESTION)
    assert generated.sql == "SELECT 1 FROM dim_campaign"
    assert provider.calls == 2


def test_the_retry_waits_the_configured_backoff() -> None:
    provider = _SequenceProvider(_unavailable(503), "SELECT 1 FROM dim_campaign")
    generator = _generator(provider, retry_backoff_seconds=0.25)
    generator.generate(QUESTION)
    assert generator.slept == [0.25]  # type: ignore[attr-defined]


def test_a_successful_first_attempt_never_waits() -> None:
    generator = _generator(_SequenceProvider("SELECT 1 FROM dim_campaign"))
    generator.generate(QUESTION)
    assert generator.slept == []  # type: ignore[attr-defined]


def test_a_503_twice_is_a_refusal_that_says_it_was_retried() -> None:
    """ "Click again" and "this one is persistent" are different instructions."""
    provider = _SequenceProvider(_unavailable(503))
    with pytest.raises(SqlGenerationFailed) as failed:
        _generator(provider).generate(QUESTION)
    assert provider.calls == 2
    assert "503" in str(failed.value)
    assert "retried once" in str(failed.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_no_status_other_than_503_is_retried(status: int) -> None:
    """A 4xx is a real answer about a real problem; delaying it helps nobody."""
    provider = _SequenceProvider(_unavailable(status))
    with pytest.raises(SqlGenerationFailed) as failed:
        _generator(provider).generate(QUESTION)
    assert provider.calls == 1, f"{status} was retried"
    assert "retried once" not in str(failed.value)


def test_a_timeout_is_not_retried() -> None:
    """A second full wait is worse for the caller than a fast refusal."""
    provider = _SequenceProvider(
        ProviderUnavailable("gemini request timed out (url REDACTED, key REDACTED)")
    )
    with pytest.raises(SqlGenerationFailed):
        _generator(provider).generate(QUESTION)
    assert provider.calls == 1


def test_an_arbitrary_exception_is_not_retried() -> None:
    provider = _SequenceProvider(RuntimeError("something else entirely"))
    with pytest.raises(SqlGenerationFailed):
        _generator(provider).generate(QUESTION)
    assert provider.calls == 1


def test_one_refusal_is_one_breaker_failure_however_many_attempts_it_took() -> None:
    """The breaker still opens after the same number of refusals a user saw.

    Recording each attempt would open it twice as fast on a merely flaky
    backend, which is the opposite of what a retry is for.
    """
    provider = _SequenceProvider(_unavailable(503))
    generator = _generator(provider, failure_threshold=2, cooldown_seconds=60.0)
    for _ in range(4):
        with pytest.raises(SqlGenerationFailed):
            generator.generate(QUESTION)
    assert provider.calls == 4, "the breaker opened after the wrong number of refusals"


def test_an_intermittent_503_never_accumulates_toward_opening() -> None:
    """A 503 that the retry recovers from is recorded as the success it was."""
    provider = _SequenceProvider(_unavailable(503), "SELECT 1 FROM dim_campaign")
    generator = _generator(provider)
    for _ in range(10):
        provider.calls = 0
        assert generator.generate(QUESTION).sql == "SELECT 1 FROM dim_campaign"


def test_a_retried_failure_still_carries_no_key_and_no_dsn() -> None:
    """The retry must not reintroduce the chain the raise site removed."""
    dsn = f"postgresql://adsignal:{SECRET}@db.internal:5432/adsignal"
    provider = _SequenceProvider(_unavailable(503), RuntimeError(f"connection to {dsn} failed"))
    with pytest.raises(SqlGenerationFailed) as failed:
        _generator(provider).generate(QUESTION)
    message = _chain(failed.value)
    assert SECRET not in message
    assert "postgresql://" not in message


def test_the_status_comes_from_the_response_not_from_the_message() -> None:
    """A retry predicate that parsed prose would break on a reworded string."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    provider = GeminiSqlProvider(api_key=SECRET, model=MODEL, client=_client(handler))
    with pytest.raises(ProviderUnavailable) as raised:
        provider.generate_sql(QUESTION)
    assert raised.value.status == 503


def test_a_transport_failure_carries_no_status_at_all() -> None:
    """`None` is what stops a timeout or a refused connection being retried."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = GeminiSqlProvider(api_key=SECRET, model=MODEL, client=_client(handler))
    with pytest.raises(ProviderUnavailable) as raised:
        provider.generate_sql(QUESTION)
    assert raised.value.status is None


def test_the_narrative_path_is_left_alone() -> None:
    """The retry is `/ask`'s. `NarrativeChain` enters a provider exactly once.

    The chain's budget gates *entry* and never cancels a running provider, so
    a retry inside the shared transport would have stretched every narrative
    provider's worst case past a budget that cannot stop it. Keeping the
    retry in `SqlGenerator` is what confines it.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    provider = GeminiProvider(api_key=SECRET, model=MODEL, client=_client(handler))
    with pytest.raises(ProviderUnavailable):
        provider.generate(NARRATIVE_REQUEST)
    assert calls["n"] == 1
