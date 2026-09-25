"""`POST /ask` — natural-language questions answered from named query templates.

**Design decision:** the spec calls this "NL→SQL". This implementation does
not generate SQL. A keyword matcher selects among the fixed set of templates
in `adsignal.api.queries` by *name*, and that name is validated against
`QUERY_TEMPLATES` before anything runs. A request supplying an unrecognised
name is rejected with a 400, never executed. Because the selector can only
ever emit one of a handful of known strings, never a query, injection is not
a risk that has been defended against — it structurally cannot occur.

**Why there is no LLM-driven template selection yet:** the existing
`NarrativeChain`/`NarrativeProvider` contract (`narrative/types.py`,
`narrative/build.py`) is generate(request: NarrativeRequest) -> str, where
`NarrativeRequest` is a closed dataclass of precomputed statistics — it has
no way to carry an arbitrary free-text question, and both of those modules
are fixed contracts from earlier tasks. Reusing "the narrative chain" for
template *selection*, as the plan's prose describes, is not expressible
without changing that contract, so this module resolves every question
through the keyword matcher below. See the task report for detail.

`/search` and this endpoint are the two that do live work per request, so
both are rate-limited (`TokenBucket`, keyed per client IP, 429 when refused)
with their own budgets. This one is additionally the only one with a response
cache (identical questions, cache hit or miss, still cost one token — the
cache saves the template resolution and database round-trip, not the
rate-limit budget; otherwise a burst of one repeated question would never
trip the limiter).
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from adsignal.api.cache import TtlCache
from adsignal.api.deps import client_key, get_ask_cache, get_ask_rate_limiter, get_settings
from adsignal.api.queries import QUERY_TEMPLATES
from adsignal.api.ratelimit import TokenBucket
from adsignal.config import Settings
from adsignal.warehouse.template_reads import run_template

router = APIRouter(tags=["ask"])

# `/ask`'s tuning, read by `create_app`, which builds the bucket and the cache
# and puts them on `app.state` (they are per-app collaborators injected through
# `deps.py`, not module-level singletons). The capacity/refill pair leaves a
# caller a burst of ten and five per second sustained; the cache is bounded as
# well as TTL'd because its key is caller-supplied free text.
ASK_RATE_LIMIT_CAPACITY = 10
ASK_RATE_LIMIT_REFILL_PER_SECOND = 5.0
ASK_CACHE_TTL_SECONDS = 30.0
ASK_CACHE_CAPACITY = 512

# Keyword -> template name, checked in order; the first match wins. This is
# the "otherwise a keyword matcher" branch described in the module docstring
# — used unconditionally today (see the note above on why the provider-chain
# branch does not exist yet).
_KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    ("cause", "top_causes"),
    ("cpa", "worst_cpa_campaigns"),
    ("cost per acquisition", "worst_cpa_campaigns"),
    ("spend", "spend_by_channel"),
    ("channel", "spend_by_channel"),
    ("anomal", "anomaly_count"),
    ("how many", "anomaly_count"),
)


class AskRequest(BaseModel):
    """`POST /ask` body."""

    question: str = Field(min_length=1)
    # An explicit template name, bypassing keyword matching entirely. Still
    # validated against `QUERY_TEMPLATES` below - this field exists so a
    # caller (or a future model-driven resolver) can name a template
    # directly, and so Review Focus 5's rejection test has something to
    # attack.
    template: str | None = None
    parameters: dict[str, int] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def _reject_blank_question(cls, value: str) -> str:
        """A whitespace-only question is a 422, not a nonsense template match."""
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class AskResponse(BaseModel):
    """`POST /ask` response."""

    template: str
    answer: str
    rows: list[dict[str, Any]]


def _match_template(question: str) -> str:
    """Pick a template name by keyword, or "unknown" if nothing matches."""
    lowered = question.lower()
    for keyword, name in _KEYWORD_RULES:
        if keyword in lowered:
            return name
    return "unknown"


def _summarize(name: str, rows: list[dict[str, Any]]) -> str:
    """Render a short natural-language answer from a template's result rows."""
    if name == "unknown":
        return "I don't have a named query that answers that question yet."
    if not rows:
        return "No matching rows were found."
    if name == "anomaly_count":
        return f"There have been {rows[0]['count']} anomalies recorded."
    if name == "top_causes":
        parts = ", ".join(f"{row['cause']} ({row['occurrences']})" for row in rows)
        return f"The most common anomaly causes are: {parts}."
    if name == "worst_cpa_campaigns":
        parts = ", ".join(f"{row['campaign_id']} (${row['cpa']:.2f})" for row in rows)
        return f"The highest cost-per-acquisition campaigns are: {parts}."
    if name == "spend_by_channel":
        parts = ", ".join(f"{row['channel']} (${row['total_spend']:.2f})" for row in rows)
        return f"Spend by channel: {parts}."
    return "Done."  # pragma: no cover - unreachable while every branch above is exhaustive


def _answer(dsn: str, template_name: str, parameters: dict[str, int]) -> AskResponse:
    """Resolve one template name to an answer, or raise the matching 4xx.

    `QueryTemplate.bind` is the only place request data reaches the query, and
    it returns already-validated values — never a string that could be
    concatenated into SQL. Execution itself belongs to `warehouse/`.
    """
    if template_name == "unknown":
        return AskResponse(template="unknown", answer=_summarize("unknown", []), rows=[])
    if template_name not in QUERY_TEMPLATES:
        raise HTTPException(status_code=400, detail=f"unknown template: {template_name!r}")

    template = QUERY_TEMPLATES[template_name]
    try:
        bound = template.bind(parameters)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    rows = run_template(dsn, template.sql, template.columns, bound)
    return AskResponse(template=template_name, answer=_summarize(template_name, rows), rows=rows)


@router.post("/ask", response_model=AskResponse)
def post_ask(
    body: AskRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    rate_limiter: TokenBucket = Depends(get_ask_rate_limiter),
    cache: TtlCache[AskResponse] = Depends(get_ask_cache),
) -> AskResponse:
    """Answer a question by running one named, parameterized query template."""
    if not rate_limiter.allow(client_key(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    # `parameters` is part of the key: two questions identical but for their
    # `limit` are different questions, and omitting it served one's answer
    # for the other. Sorted so dict ordering cannot split one logical key.
    bound_key = ",".join(f"{name}={value}" for name, value in sorted(body.parameters.items()))
    cache_key = f"{body.template or ''}|{body.question.strip().lower()}|{bound_key}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    template_name = body.template if body.template is not None else _match_template(body.question)
    response = _answer(settings.postgres_dsn, template_name, body.parameters)
    cache.put(cache_key, response)
    return response
