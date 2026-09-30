"""`GET /campaigns/{campaign_id}/summary` — prose about one campaign's run.

The stored daily narrative describes one day across every campaign. A reader
looking at a single campaign was therefore reading about something else, and
had no way to ask "how has this one done". This writes that.

Generated on request rather than nightly, and deliberately so: there are
hundreds of campaigns, almost all of them are never looked at, and writing a
paragraph for each every night would spend a model call per campaign per day
to answer a question nobody asked. The cost of on-demand is a few seconds
after a click, which is the right trade for a dashboard.

Like `/ask`, this is rate limited per client and cached, because both bound
the same finite thing: a free-tier model quota. Unlike `/ask`, it always has
something to say — with no key configured the facts are written up locally,
so the endpoint degrades to a deterministic paragraph rather than an error.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from adsignal.api.cache import TtlCache
from adsignal.api.client_ip import client_key
from adsignal.api.deps import (
    get_settings,
    get_summary_cache,
    get_summary_rate_limiter,
)
from adsignal.api.ratelimit import TokenBucket
from adsignal.api.schemas import CampaignSummaryOut
from adsignal.config import Settings
from adsignal.narrative.campaign_summary import (
    CampaignFacts,
    build_campaign_prompt,
    summarize_campaign,
)
from adsignal.narrative.providers.gemini import generate_text
from adsignal.warehouse.campaign_summary_reads import read_campaign_facts

router = APIRouter(tags=["campaigns"])

# Tighter than `/ask`'s burst of ten. A summary costs a model call and cannot
# be served from a shared cache the way a repeated question can, since the key
# is a campaign id and there are hundreds of them. Five in a burst is enough
# for a reader clicking between campaigns and far short of what would drain a
# free-tier quota.
SUMMARY_RATE_LIMIT_CAPACITY = 5
SUMMARY_RATE_LIMIT_REFILL_PER_SECOND = 0.5
# Long enough that clicking back to a campaign is instant, short enough that a
# summary follows the data within one nightly run.
SUMMARY_CACHE_TTL_SECONDS = 600.0
SUMMARY_CACHE_CAPACITY = 256

#: The provider name reported when the summary was composed locally. Matches
#: the daily narrative's vocabulary so the dashboard can say "written by" the
#: same way on both.
LOCAL_PROVIDER = "stub"


def _written_by_model(facts: CampaignFacts, settings: Settings) -> tuple[str, str] | None:
    """Return `(text, provider)` from the model, or `None` to fall back.

    Every failure returns `None` rather than raising: a summary is a nicety
    and the local one is genuinely good, so a model outage must degrade the
    paragraph rather than the page. Nothing about the failure reaches the
    caller, because a provider exception can carry a key-bearing URL.
    """
    key = settings.gemini_api_key.get_secret_value()
    if not key:
        return None
    try:
        text = generate_text(
            build_campaign_prompt(facts),
            api_key=key,
            model=settings.gemini_model,
            timeout=settings.narrative_timeout_seconds,
        )
    except Exception:  # noqa: BLE001 - any provider failure falls back to local prose
        return None
    return (text.strip(), "gemini") if text.strip() else None


@router.get("/campaigns/{campaign_id}/summary", response_model=CampaignSummaryOut)
def get_campaign_summary(
    campaign_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
    rate_limiter: TokenBucket = Depends(get_summary_rate_limiter),
    cache: TtlCache[Any] = Depends(get_summary_cache),
) -> CampaignSummaryOut:
    """Summarise one campaign's whole run. 404 when it has no daily rows."""
    if not rate_limiter.allow(client_key(request, settings.trusted_proxy_hops)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    # Checked before the cache: a 404 must stay a 404 rather than being
    # served a neighbouring campaign's paragraph from a stale entry.
    facts = read_campaign_facts(settings.postgres_dsn, campaign_id)
    if facts is None:
        raise HTTPException(status_code=404, detail=f"no daily rows for {campaign_id}")

    cached: CampaignSummaryOut | None = cache.get(campaign_id)
    if cached is not None:
        return cached

    written = _written_by_model(facts, settings)
    text, provider = written if written is not None else (summarize_campaign(facts), LOCAL_PROVIDER)

    response = CampaignSummaryOut(
        campaign_id=facts.campaign_id,
        campaign_name=facts.campaign_name,
        text=text,
        provider=provider,
        first_day=facts.first_day,
        last_day=facts.last_day,
        days=facts.days,
        spend=facts.spend,
        conversions=facts.conversions,
        flagged_days=facts.flagged_days,
        escalated_days=facts.escalated_days,
    )
    cache.put(campaign_id, response)
    return response
