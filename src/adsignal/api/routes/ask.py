"""`POST /ask` — natural-language questions answered by generated SQL.

This endpoint used to match a question against four named query templates.
That version could not be injected into, structurally: its selector emitted
one of four fixed strings and never a query. It also could not answer
anything its author had not thought of, which is what the endpoint is for.

It now sends the question and a description of seven tables to Gemini and
runs the SQL that comes back. **The structural guarantee is gone, and three
independent controls replace it**, each sufficient on its own:

1. A read-only Postgres role (`adsignal_ro`). Generated SQL runs only on
   `settings.postgres_readonly_dsn`; the admin DSN is never passed to the
   executor. The role holds SELECT on the seven allowlisted tables, owns
   nothing, and is explicitly revoked on `model_artifacts` and
   `entity_embeddings`.
2. A read-only transaction with a 5 s statement timeout, so a write slips
   past nothing and a runaway query dies on its own.
3. A pure validator (`warehouse.sql_guard`), which rejects multiple
   statements, comments, every writing or administrative keyword, any table
   outside the allowlist, and any LIMIT above 200.

The prompt asks for the same shape. That is a quality measure and nothing
more: a model that ignores every word of it produces a refusal, not a
breach. `tests/test_ask_reads.py` proves the point by handing the executor a
`DROP TABLE` the validator never saw.

The generated SQL is returned to the caller on purpose. A reader who can see
the query can judge the answer; one who cannot has to trust it.

The prompt also carries the period the warehouse covers, read off
`fct_campaign_daily` by `warehouse.ask_reads.load_date_coverage`. Without it
the model anchored "last month" to `CURRENT_DATE` and filtered a corpus that
ends in the past down to nothing — a correct query and an empty answer to an
ordinary question. The same range explains an empty answer after the fact,
through `AskResponse.notice`.

**This endpoint answers 200 or it answers 429 or 422, and nothing else.** A
rate-limit refusal and a malformed body are the two status codes; everything
past that point — a model failure, a refused query, a database error, and
anything nobody anticipated — is a 200 carrying the same eight fields. The
catch-all around `_answer` is what makes that last clause true rather than
aspirational, and it exists because the failure that broke it was a perfectly
ordinary one: SQL truncated at a token limit.

Rate limiting and caching are unchanged from the template version: a
`TokenBucket` keyed per client IP (429 when refused) and a bounded TTL cache
keyed on the question. A cache hit still costs a token — otherwise a burst of
one repeated question would never trip the limiter — and it saves the model
call and the database round trip, which are the expensive parts.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from adsignal.api.cache import TtlCache
from adsignal.api.client_ip import client_key
from adsignal.api.deps import (
    get_ask_cache,
    get_ask_rate_limiter,
    get_settings,
    get_sql_generator,
)
from adsignal.api.ratelimit import TokenBucket
from adsignal.config import Settings
from adsignal.narrative.sqlgen import SqlGenerationFailed, SqlGenerator, SqlQuestion
from adsignal.warehouse.ask_reads import QueryRefused, load_date_coverage, run_readonly_query
from adsignal.warehouse.ask_schema import describe_schema, empty_result_notice
from adsignal.warehouse.sql_guard import SqlRejected, validate_sql

router = APIRouter(tags=["ask"])

# `/ask`'s tuning, read by `create_app`, which builds the bucket and the cache
# and puts them on `app.state` (they are per-app collaborators injected through
# `deps.py`, not module-level singletons). A request now costs a model call as
# well as a query, so the burst of ten and five per second sustained is if
# anything more generous than it should be; the cache is bounded as well as
# TTL'd because its key is caller-supplied free text.
ASK_RATE_LIMIT_CAPACITY = 10
ASK_RATE_LIMIT_REFILL_PER_SECOND = 5.0
ASK_CACHE_TTL_SECONDS = 30.0
ASK_CACHE_CAPACITY = 512

# The refusal rule reported when the deployment has no model or no read-only
# database configured. Distinct from every validator and database rule: it
# says the endpoint is switched off, not that the question was bad.
NOT_CONFIGURED_RULE = "not_configured"

# The refusal reported when something raised that this module did not expect.
# The endpoint's contract is that it always answers 200 with the same eight
# fields, and a contract that holds only for the failures someone anticipated
# is not a contract. A truncated query — what a model emits when it hits its
# token limit — reached `sqlglot.TokenError`, which is a sibling of
# `ParseError` rather than a subclass, and became a 500; a deeply nested one
# raises `RecursionError`, which is not a sqlglot error at all. Both are now
# refusals, and so is whatever the next one turns out to be.
UNEXPECTED_RULE = "unexpected_error"


class AskRequest(BaseModel):
    """`POST /ask` body. A question, and nothing else.

    The template version also took a `template` name and a `parameters`
    dict. Both are gone: there are no templates to name, and every value the
    question implies is now the model's problem and the validator's.
    """

    question: str = Field(min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def _reject_blank_question(cls, value: str) -> str:
        """A whitespace-only question is a 422, not a prompt sent to a model."""
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class AskResponse(BaseModel):
    """`POST /ask` response, identical in shape whether answered or refused.

    One shape, always: a client renders `refusal` when it is present and the
    table when it is not, and never has to branch on a status code. `sql` is
    populated whenever a query was generated — including when it was then
    refused, because seeing the rejected query is how a reader judges the
    refusal. `notice` is the one field that is neither an answer nor a
    refusal; see its own docstring.
    """

    question: str
    sql: str | None
    columns: list[str]
    rows: list[dict[str, Any]]
    provider: str | None
    refusal: str | None
    rule: str | None
    notice: str | None = None
    """Context for an answer that is correct but unhelpful on its own.

    Present today for exactly one case: a query that filtered on
    `event_date` and matched nothing, where the reason is very often that the
    filter fell outside what the warehouse holds. `rows: []` is true and
    tells a reader nothing; this says what the data actually covers.

    It belongs here rather than in the dashboard because the API is the side
    that knows: it has just read the range off the warehouse to ground the
    prompt, and a client would otherwise need a second endpoint and a second
    round trip to say the same sentence. It is deliberately **not** a
    `refusal` - nothing was refused, and a client that renders refusals as
    errors must not render this as one.

    Defaulted to `None` so every refusal path constructs unchanged.
    """


def _not_configured(question: str, reason: str) -> AskResponse:
    """The honest answer when this deployment cannot run the endpoint.

    A 200 with an empty result and a plain explanation, not a 500 and not a
    fabricated answer: "no key is configured" is a fact about the
    deployment, and the dashboard renders it as one.
    """
    return AskResponse(
        question=question,
        sql=None,
        columns=[],
        rows=[],
        provider=None,
        refusal=reason,
        rule=NOT_CONFIGURED_RULE,
    )


def _refused(question: str, sql: str | None, provider: str, reason: str, rule: str) -> AskResponse:
    """A refusal: the query that was rejected, which rule rejected it, no rows."""
    return AskResponse(
        question=question,
        sql=sql,
        columns=[],
        rows=[],
        provider=provider,
        refusal=reason,
        rule=rule,
    )


def _answer(question: str, generator: SqlGenerator, readonly_dsn: str) -> AskResponse:
    """Generate, validate and run one query, or return the refusal that stopped it.

    Each stage can only refuse; none of them can widen what the next one
    accepts. The SQL reported is always the exact string handed to the
    database — `validate_sql` returns what may run, so the query a reader
    sees and the query that ran cannot differ.
    """
    # Read before the model is called, because it is what the model is told:
    # without it Gemini writes correct `CURRENT_DATE`-relative SQL over a
    # corpus that ended in the past. Cached per DSN inside `load_date_coverage`,
    # so this is not a round trip per question.
    coverage = load_date_coverage(readonly_dsn)

    try:
        generated = generator.generate(
            SqlQuestion(question=question, schema_description=describe_schema(coverage))
        )
    except SqlGenerationFailed as failure:
        return _refused(question, None, generator.name, str(failure), "generation_failed")

    try:
        safe_sql = validate_sql(generated.sql)
    except SqlRejected as rejection:
        return _refused(
            question, generated.sql, generated.provider, rejection.reason, rejection.rule
        )

    try:
        result = run_readonly_query(readonly_dsn, safe_sql)
    except QueryRefused as refusal:
        return _refused(question, safe_sql, generated.provider, refusal.reason, refusal.rule)

    return AskResponse(
        question=question,
        sql=safe_sql,
        columns=list(result.columns),
        rows=result.rows,
        provider=generated.provider,
        refusal=None,
        rule=None,
        notice=None if result.rows else empty_result_notice(safe_sql, coverage),
    )


@router.post("/ask", response_model=AskResponse)
def post_ask(
    body: AskRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    rate_limiter: TokenBucket = Depends(get_ask_rate_limiter),
    cache: TtlCache[AskResponse] = Depends(get_ask_cache),
    generator: SqlGenerator | None = Depends(get_sql_generator),
) -> AskResponse:
    """Answer a question by generating SQL, validating it, and running it read-only."""
    if not rate_limiter.allow(client_key(request, settings.trusted_proxy_hops)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    question = body.question.strip()
    cache_key = question.lower()
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if generator is None:
        return _not_configured(question, "ask is not configured — no language model is available")
    if not settings.postgres_readonly_dsn:
        return _not_configured(
            question, "ask is not configured — no read-only database connection is available"
        )

    try:
        response = _answer(question, generator, settings.postgres_readonly_dsn)
    except Exception:  # noqa: BLE001 - the contract is "always 200"; see UNEXPECTED_RULE
        # Deliberately says nothing about what raised. The exception's own
        # text can carry a DSN, a driver message or the failing statement,
        # and this response is rendered to whoever asked the question.
        return _refused(
            question,
            None,
            generator.name,
            "something went wrong answering that; try rephrasing it",
            UNEXPECTED_RULE,
        )
    cache.put(cache_key, response)
    return response
