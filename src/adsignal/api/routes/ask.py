"""`POST /ask` — natural-language questions answered from named query templates.

**Design decision:** the spec calls this "NL→SQL". This implementation does
not generate SQL. A model — or, with no provider key configured, a keyword
matcher — selects among the fixed set of templates in `adsignal.api.queries`
by *name*, and that name is validated against `QUERY_TEMPLATES` before
anything runs. A request supplying an unrecognised name is rejected with a
400, never executed. Because a model here can only ever emit one of a
handful of known strings, never a query, injection is not a risk that has
been defended against — it structurally cannot occur.

**Why there is no LLM-driven template selection yet:** the existing
`NarrativeChain`/`NarrativeProvider` contract (`narrative/types.py`,
`narrative/build.py`) is generate(request: NarrativeRequest) -> str, where
`NarrativeRequest` is a closed dataclass of precomputed statistics — it has
no way to carry an arbitrary free-text question, and both of those modules
are fixed contracts from earlier tasks. Reusing "the narrative chain" for
template *selection*, as the plan's prose describes, is not expressible
without changing that contract, so this module resolves every question
through the keyword matcher below. See the task report for detail.

This is the only endpoint in the API that does live work per request, so it
is the only one rate-limited (`TokenBucket`, keyed per client IP, 429 when
refused) and the only one with a response cache (identical questions, cache
hit or miss, still cost one token — the cache saves the template resolution
and database round-trip, not the rate-limit budget; otherwise a burst of one
repeated question would never trip the limiter).
"""

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from adsignal.api.deps import get_settings
from adsignal.api.queries import QUERY_TEMPLATES, run_template
from adsignal.api.ratelimit import TokenBucket
from adsignal.config import Settings

router = APIRouter(tags=["ask"])

# A module-level singleton: one budget and one cache shared across every
# request this process serves, mirroring how `create_app` shares a single
# `EmbeddingModel` rather than building one per request.
_BUCKET = TokenBucket(capacity=10, refill_per_second=5.0)
_CACHE_TTL_SECONDS = 30.0
_cache: dict[str, tuple[float, "AskResponse"]] = {}

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


def _client_key(request: Request) -> str:
    """The rate-limit and cache key: the caller's IP, or a fixed fallback."""
    return request.client.host if request.client is not None else "unknown"


@router.post("/ask", response_model=AskResponse)
def post_ask(
    body: AskRequest, request: Request, settings: Settings = Depends(get_settings)
) -> AskResponse:
    """Answer a question by running one named, parameterized query template."""
    if not _BUCKET.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    cache_key = f"{body.template or ''}|{body.question.strip().lower()}"
    now = time.monotonic()
    cached = _cache.get(cache_key)
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    template_name = body.template if body.template is not None else _match_template(body.question)

    if template_name == "unknown":
        response = AskResponse(template="unknown", answer=_summarize("unknown", []), rows=[])
    elif template_name not in QUERY_TEMPLATES:
        raise HTTPException(status_code=400, detail=f"unknown template: {template_name!r}")
    else:
        template = QUERY_TEMPLATES[template_name]
        try:
            rows = run_template(settings.postgres_dsn, template, body.parameters)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        response = AskResponse(
            template=template_name, answer=_summarize(template_name, rows), rows=rows
        )

    _cache[cache_key] = (now, response)
    return response
