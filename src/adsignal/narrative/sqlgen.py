"""Turning a free-text question into candidate SQL.

**Why this is a sibling of `NarrativeProvider` rather than an extension of
it.** `NarrativeRequest` is a closed dataclass of precomputed statistics,
and its docstring gives the reason: the prompt is built from it, so nothing
unexpected can reach a provider. A question is arbitrary caller text. Adding
a `question: str | None` field would have widened that guarantee's hole for
every existing caller of a contract three other providers already implement,
to express a request that shares none of its fields. `SqlProvider` is a
second, equally small contract instead — one method, one closed request —
and the two share the Gemini transport, the circuit breaker and the
redaction discipline, which is where the actual reuse was.

Deliberately free of database access: this module turns a question and a
schema *description* into a string. It never connects, never sees a row and
never learns a DSN. Executing what it produces is `warehouse.ask_reads`'s
job, and validating it first is `warehouse.sql_guard`'s.
"""

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from adsignal.config import Settings
from adsignal.narrative.breaker import CircuitBreaker
from adsignal.narrative.types import ProviderUnavailable

# The model is asked for the same shape the validator accepts. This is a
# quality measure, not a control: every rule named here is independently
# enforced by `warehouse.sql_guard`, and the read-only role enforces the
# important ones again. A model that ignores all of it produces a refusal,
# not a breach.
_INSTRUCTIONS = """\
You write PostgreSQL for a read-only analytics API. Answer with one SQL
query and nothing else: no explanation, no markdown fence, no semicolon.

Rules the query must follow, all of which are checked before it runs:
- a single SELECT (a leading WITH is fine); never INSERT, UPDATE, DELETE or
  any DDL
- read only the tables described below, by their bare names
- no SQL comments
- include a LIMIT of 200 or fewer
- if the question cannot be answered from these tables, return a query that
  answers the closest thing that can be
"""


@dataclass(frozen=True, slots=True)
class SqlQuestion:
    """Everything a provider is given: the question and the table shapes.

    Closed, like `NarrativeRequest`, and for the same reason. There is no
    field here through which a DSN, a key or a row of data could travel to a
    third party, because there is no such field at all.
    """

    question: str
    schema_description: str


@dataclass(frozen=True, slots=True)
class GeneratedSql:
    """One candidate query and the provider that wrote it."""

    sql: str
    provider: str


@runtime_checkable
class SqlProvider(Protocol):
    """One backend capable of turning a question into a candidate query."""

    name: str

    def generate_sql(self, request: SqlQuestion) -> str:
        """Return SQL, or raise if this provider cannot."""
        ...


class SqlGenerationFailed(RuntimeError):
    """The configured provider could not produce a query.

    Carries a short reason only, composed by `_failure_reason`: either the
    provider's own redacted message or, for anything else, an exception type
    name. This type exists so `api.routes.ask` can tell "the model failed"
    from "the query was refused" without inspecting exception text.
    """


class SqlGenerator:
    """One provider, guarded by a circuit breaker.

    A single provider rather than a `NarrativeChain`: Gemini is the only
    backend that answers this, and a chain of one adds an ordering and a
    budget that would describe nothing. The breaker is the part worth
    reusing — a dead backend stops costing every caller a round trip, and
    `/ask` is rate-limited per IP but not per backend.
    """

    def __init__(
        self,
        provider: SqlProvider,
        failure_threshold: int = 2,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._provider = provider
        self._breaker = CircuitBreaker(
            failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds
        )

    @property
    def name(self) -> str:
        """The provider's name, as the response reports it."""
        return self._provider.name

    def generate(self, request: SqlQuestion) -> GeneratedSql:
        """Return candidate SQL, or raise `SqlGenerationFailed`.

        The failure is composed inside the `except` block and raised *below*
        it, the same discipline `providers.gemini.generate_text` follows and
        for the same reason: `raise ... from None` clears `__cause__` but
        leaves `__context__` pointing at the provider's own exception, and a
        structured logger or an error reporter walks that chain. The
        provider's exception can be an httpx error carrying the key in a URL
        or a driver error carrying a DSN, so the only chain safe to hand on
        is no chain at all. Raising with nothing being handled is what makes
        `__context__` genuinely `None`.
        """
        if self._breaker.is_open():
            raise SqlGenerationFailed(f"{self._provider.name} is unavailable")

        failure: str | None = None
        sql = ""
        try:
            sql = self._provider.generate_sql(request)
        except Exception as error:  # noqa: BLE001 - any provider failure is one failure
            self._breaker.record_failure()
            failure = _failure_reason(self._provider.name, error)

        if failure is not None:
            raise SqlGenerationFailed(failure)
        self._breaker.record_success()
        return GeneratedSql(sql=sql, provider=self._provider.name)


def _failure_reason(provider_name: str, error: Exception) -> str:
    """What a user is told when a provider did not produce a query.

    A `ProviderUnavailable` is passed through whole. Its message was composed
    by the provider with the URL and the key already replaced by the word
    REDACTED (see `narrative.types.ProviderUnavailable`), so it can name the
    status code — and a transient 503, a rate limit, an expired key and a
    retired model are four different problems that used to reach the user as
    the single word "RuntimeError".

    Anything else contributes its type name and nothing more. No other
    exception in this process has had its text proven free of a key, a DSN or
    a URL, and an httpx error carries the request URL with the key inside it.
    """
    if isinstance(error, ProviderUnavailable):
        return str(error)
    return f"{provider_name} could not answer ({type(error).__name__})"


def build_sql_generator(settings: Settings) -> SqlGenerator | None:
    """Return the configured generator, or `None` when no key is configured.

    `None` is a state, not a failure: with `ADSIGNAL_GEMINI_API_KEY` unset —
    which is how this repository runs today — `/ask` answers 200 saying so.
    Raising here, or returning something that pretends to generate, would
    turn a plain configuration fact into either a 500 or a made-up answer.

    `get_secret_value()` is called once, here, and the raw key goes straight
    into the provider; it is never logged, stored or returned.
    """
    # Imported here rather than at module scope: the provider imports this
    # module for `build_sql_prompt` and `SqlQuestion`, so importing it at the
    # top would be a cycle.
    from adsignal.narrative.providers.gemini_sql import GeminiSqlProvider

    key = settings.gemini_api_key.get_secret_value()
    if not key:
        return None
    return SqlGenerator(
        GeminiSqlProvider(
            key, model=settings.gemini_model, timeout=settings.narrative_timeout_seconds
        )
    )


def build_sql_prompt(request: SqlQuestion) -> str:
    """Return the prompt: the rules, the schema, and the question.

    Contains no key, no DSN, no connection string and no row of data — the
    caller passes a schema *description* (`warehouse.ask_schema`), which is
    table and column names only.
    """
    return f"{_INSTRUCTIONS}\nTables:\n{request.schema_description}\nQuestion: {request.question}\n"


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(?P<body>.*?)\n?```\s*$", re.DOTALL)


def extract_sql(text: str) -> str:
    """Return the SQL inside a model's answer, unwrapping a markdown fence.

    Models fence code even when told not to. A fence is not SQL, so stripping
    it here is the difference between a working answer and a refusal for
    syntax the user did not write. Nothing else is rewritten: what comes out
    of here is what the validator sees and what the user is shown.
    """
    stripped = text.strip()
    fenced = _FENCE_RE.match(stripped)
    if fenced is not None:
        return fenced.group("body").strip()
    return stripped
