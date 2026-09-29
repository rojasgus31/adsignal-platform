"""`POST /ask` end to end: the no-key state, an answer, and the refusals.

The stub generator here stands in for Gemini. Its job in most of these tests
is to be hostile — to return exactly the SQL a compromised or confused model
would return — so that what is tested is the endpoint's behaviour when the
model misbehaves, which is the only interesting case.
"""

import os
from typing import Any

import psycopg
import pytest

from adsignal.narrative.sqlgen import SqlGenerator, SqlQuestion

# Same pattern as every other DSN-driven test module: the env var is an
# override, not a requirement.
TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)

SECRET = "sk-super-secret-key-do-not-leak"


class _FixedProvider:
    """A `SqlProvider` that returns whatever SQL the test names."""

    def __init__(self, sql: str) -> None:
        self.name = "stub"
        self.sql = sql
        self.calls = 0

    def generate_sql(self, request: SqlQuestion) -> str:
        self.calls += 1
        return self.sql


def _install(client: Any, sql: str) -> _FixedProvider:
    """Point this app's `/ask` at a generator that always returns `sql`."""
    provider = _FixedProvider(sql)
    client.app.state.sql_generator = SqlGenerator(provider)
    return provider


def _ask(client: Any, question: str) -> dict[str, Any]:
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_no_model_configured_answers_two_hundred_and_says_so(client) -> None:
    """The state this repository is in today: `ADSIGNAL_GEMINI_API_KEY` unset.

    It must be a rendered sentence in the dashboard, not a 500 and not an
    invented answer.
    """
    client.app.state.sql_generator = None
    body = _ask(client, "how many anomalies were there?")
    assert body["rule"] == "not_configured"
    assert "not configured" in body["refusal"]
    assert body["sql"] is None
    assert body["provider"] is None
    assert body["rows"] == []
    assert body["columns"] == []


def test_no_readonly_dsn_configured_also_degrades_honestly(client) -> None:
    """A key but no read-only database is equally unconfigured, and says so."""
    _install(client, "SELECT 1 FROM dim_campaign")
    body = _ask(client, "anything at all")
    assert body["rule"] == "not_configured"
    assert "read-only" in body["refusal"]
    assert body["rows"] == []


def test_a_generated_query_is_run_and_shown(ask_client) -> None:
    _install(ask_client, "SELECT campaign_id, channel FROM dim_campaign LIMIT 3")
    body = _ask(ask_client, "list a few campaigns")
    assert body["refusal"] is None
    assert body["rule"] is None
    assert body["provider"] == "stub"
    assert body["sql"] == "SELECT campaign_id, channel FROM dim_campaign LIMIT 3"
    assert body["columns"] == ["campaign_id", "channel"]
    assert 0 < len(body["rows"]) <= 3
    assert set(body["rows"][0]) == {"campaign_id", "channel"}


def test_the_sql_shown_is_the_sql_that_ran(ask_client) -> None:
    """The cap is appended before execution, so the user sees the real query."""
    _install(ask_client, "SELECT campaign_id FROM dim_campaign")
    body = _ask(ask_client, "every campaign")
    assert body["sql"] == "SELECT campaign_id FROM dim_campaign LIMIT 200"
    assert len(body["rows"]) <= 200


def test_a_hostile_model_answer_is_refused_and_the_table_survives(ask_client) -> None:
    """The endpoint's worst case: the model returns a stacked DROP."""
    _install(ask_client, "SELECT 1 FROM dim_campaign; DROP TABLE dim_campaign")
    body = _ask(ask_client, "drop everything")
    assert body["rule"] == "single_statement"
    assert body["sql"] == "SELECT 1 FROM dim_campaign; DROP TABLE dim_campaign"
    assert body["rows"] == []

    with psycopg.connect(TEST_DSN) as conn:
        row = conn.execute("SELECT to_regclass('dim_campaign')").fetchone()
    assert row is not None and row[0] is not None, "the table was dropped"


def test_a_query_against_an_excluded_table_is_refused(ask_client) -> None:
    _install(ask_client, "SELECT * FROM model_artifacts LIMIT 5")
    body = _ask(ask_client, "show me the pickles")
    assert body["rule"] == "table_allowlist"
    assert "model_artifacts" in body["refusal"]


def test_a_refused_query_is_still_shown_to_the_reader(ask_client) -> None:
    """Seeing the rejected SQL is how a reader judges the refusal."""
    _install(ask_client, "SELECT pg_read_file('/etc/passwd')")
    body = _ask(ask_client, "read a file")
    assert body["sql"] == "SELECT pg_read_file('/etc/passwd')"
    assert body["rule"] == "forbidden_keyword"


def test_a_model_failure_is_a_refusal_not_a_five_hundred(ask_client) -> None:
    class _Broken:
        name = "stub"

        def generate_sql(self, request: SqlQuestion) -> str:
            raise RuntimeError("gemini returned 503 (url REDACTED, key REDACTED)")

    ask_client.app.state.sql_generator = SqlGenerator(_Broken())
    body = _ask(ask_client, "anything")
    assert body["rule"] == "generation_failed"
    assert body["sql"] is None
    assert body["provider"] == "stub"


def test_a_refusal_never_leaks_a_dsn_or_a_key(ask_client) -> None:
    _install(ask_client, "DELETE FROM dim_campaign")
    response = ask_client.post("/ask", json={"question": "delete it all"})
    assert response.status_code == 200
    text = response.text
    assert "postgresql://" not in text
    assert "adsignal_ro" not in text
    assert SECRET not in text

    # The key this app is really configured with (`conftest.ask_client` gives
    # it a sentinel). Asserting only against a `SECRET` the app was never
    # given is an assertion that cannot fail, which is what this test did
    # before fix round 1. If the fixture ever stops configuring a key this
    # line fails loudly rather than passing vacuously, because `"" not in
    # text` is never true.
    configured = ask_client.app.state.settings.gemini_api_key.get_secret_value()
    assert configured not in text
    assert "Traceback" not in text


def test_an_identical_question_is_served_from_the_cache(ask_client) -> None:
    """The cache saves the model call, which is now the expensive part."""
    provider = _install(ask_client, "SELECT campaign_id FROM dim_campaign LIMIT 1")
    first = _ask(ask_client, "one campaign please")
    second = _ask(ask_client, "One Campaign Please")
    assert first == second
    assert provider.calls == 1, "the second identical question called the model again"


def test_ask_is_still_rate_limited_per_ip(ask_client) -> None:
    _install(ask_client, "SELECT campaign_id FROM dim_campaign LIMIT 1")
    statuses = [
        ask_client.post("/ask", json={"question": f"question number {index}"}).status_code
        for index in range(30)
    ]
    assert 429 in statuses, "no request was ever rate-limited"
    assert statuses[0] == 200


def test_a_blank_question_is_rejected_before_any_model_is_called(client) -> None:
    provider = _install(client, "SELECT 1")
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert provider.calls == 0


def test_an_overlong_question_is_rejected(client) -> None:
    assert client.post("/ask", json={"question": "x" * 5000}).status_code == 422


@pytest.mark.parametrize(
    "field", ["question", "sql", "columns", "rows", "provider", "refusal", "rule"]
)
def test_every_documented_field_is_always_present(ask_client, field: str) -> None:
    """The dashboard client is hand-written from this shape; it must not vary."""
    _install(ask_client, "SELECT campaign_id FROM dim_campaign LIMIT 1")
    assert field in _ask(ask_client, f"a question about {field}")


def test_an_unrestricted_readonly_dsn_refuses_rather_than_running(ask_client) -> None:
    """Fix round 1, finding 3, through the endpoint.

    Pointing `postgres_readonly_dsn` at the admin DSN is the misconfiguration
    that silently deletes control 1. `/ask` must refuse to answer at all
    rather than run generated SQL as an unrestricted role.
    """
    from adsignal.warehouse.ask_reads import forget_verified_dsns

    settings = ask_client.app.state.settings
    original = settings.postgres_readonly_dsn
    settings.postgres_readonly_dsn = settings.postgres_dsn
    forget_verified_dsns()
    try:
        _install(ask_client, "SELECT campaign_id FROM dim_campaign LIMIT 1")
        body = _ask(ask_client, "anything at all")
    finally:
        settings.postgres_readonly_dsn = original
        forget_verified_dsns()

    assert body["rule"] == "readonly_dsn_unverified"
    assert body["rows"] == []
    assert "postgresql://" not in body["refusal"]


# Fix round 5. The contract is 200-or-429-or-422. Nothing else.


def test_truncated_sql_from_the_model_is_a_refusal_not_a_500(ask_client) -> None:
    """The likeliest real failure: the model hit its token limit mid-literal.

    This returned a 500 before fix round 5, breaking both the seven-field
    contract and the "never a 500" guarantee.
    """
    _install(ask_client, "SELECT * FROM dim_campaign WHERE channel = 'sea")
    body = _ask(ask_client, "which channel?")
    assert body["rule"] == "invalid_query"
    assert body["rows"] == []
    assert body["sql"] == "SELECT * FROM dim_campaign WHERE channel = 'sea"


def test_an_unexpected_exception_is_still_a_two_hundred(ask_client) -> None:
    """The catch-all, tested through something no rule anticipates."""

    class _Exploding:
        name = "stub"

        def generate_sql(self, request: SqlQuestion) -> str:
            raise MemoryError("not enough memory")

    ask_client.app.state.sql_generator = SqlGenerator(_Exploding())
    body = _ask(ask_client, "anything")
    assert body["rule"] in ("generation_failed", "unexpected_error")
    assert body["rows"] == []


def test_a_deeply_nested_query_is_a_refusal_not_a_500(ask_client) -> None:
    """Deep nesting raises `RecursionError`, which is not a sqlglot error."""
    _install(ask_client, "SELECT " + "(" * 400 + "1" + ")" * 400 + " FROM dim_campaign")
    body = _ask(ask_client, "deeply nested")
    assert body["refusal"] is not None
    assert body["rows"] == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM dim_campaign WHERE channel = 'sea",
        "SELECT campaign_id FROM dim_campaign /* unfinished",
        "",
        "not sql at all",
        "SELECT * FROM model_artifacts",
        "DROP TABLE dim_campaign",
        "SELECT " + "(" * 400 + "1" + ")" * 400,
    ],
)
def test_every_malformed_answer_keeps_the_seven_field_contract(ask_client, sql: str) -> None:
    _install(ask_client, sql)
    response = ask_client.post("/ask", json={"question": f"q {len(sql)}"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"question", "sql", "columns", "rows", "provider", "refusal", "rule"}
    assert body["refusal"] is not None
