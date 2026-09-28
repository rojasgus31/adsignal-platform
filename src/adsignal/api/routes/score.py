"""`POST /score` — score one campaign-day on demand.

This is the endpoint that proves the model is loaded and serving rather than
that a metrics row was once written to a table. `GET /anomalies` returns what
the nightly batch decided; this runs the same two estimators, in this
process, against a feature row built now.

Three refusals matter more than the number it returns:

* An unknown campaign-day is a 404. So is a campaign-day too young to have
  the seven prior days its rolling baselines are computed from — with NaN
  baselines the detector still returns a confident-looking score, and a
  confidently wrong score is worse than a refusal. `warehouse.feature_reads`
  decides which, and writes the reason.
* A deployment with no trained artifacts is a 503, not a zero.
* It is rate-limited like `/search`, per caller IP, because it does even more
  work per call: a bounded warehouse read plus three ensemble forward passes.

`api/` runs models here — `/search` already does — but issues no SQL: the
feature row is read through `warehouse/` like every other endpoint's data.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from adsignal.api.client_ip import client_key
from adsignal.api.deps import get_score_rate_limiter, get_scoring_models, get_settings
from adsignal.api.model_loader import ScoringModelHolder
from adsignal.api.ratelimit import TokenBucket
from adsignal.api.schemas import ScoreOut, ScoreRequest
from adsignal.config import Settings
from adsignal.ml.scoring import ScoreResult
from adsignal.warehouse.feature_reads import FeatureRow, read_feature_row

router = APIRouter(tags=["score"])

# Tighter than `/search`'s 5-burst / 1-per-second: a call here runs three
# ensemble forward passes on top of a warehouse read. Read by `create_app`,
# which builds the bucket and puts it on `app.state`.
SCORE_RATE_LIMIT_CAPACITY = 5
SCORE_RATE_LIMIT_REFILL_PER_SECOND = 0.5


@router.post("/score", response_model=ScoreOut)
def post_score(
    body: ScoreRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    models: ScoringModelHolder = Depends(get_scoring_models),
    rate_limiter: TokenBucket = Depends(get_score_rate_limiter),
) -> ScoreOut:
    """Score one campaign-day now, with the models this process has loaded."""
    if not rate_limiter.allow(client_key(request, settings.trusted_proxy_hops)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    resolution = models.resolve()
    if resolution.models is None:
        raise HTTPException(status_code=503, detail=resolution.refusal_reason)
    loaded = resolution.models

    row = read_feature_row(settings.postgres_dsn, body.campaign_id, body.event_date)
    if not isinstance(row, FeatureRow):
        raise HTTPException(status_code=404, detail=row.reason)

    return _to_score_out(body, loaded.score(row.frame))


def _to_score_out(body: ScoreRequest, result: ScoreResult) -> ScoreOut:
    """Translate the ML layer's verdict into the API's DTO."""
    return ScoreOut(
        campaign_id=body.campaign_id,
        event_date=body.event_date,
        anomaly_score=result.anomaly_score,
        is_anomaly=result.is_anomaly,
        cause=result.cause,
        confidence=result.confidence,
        severity=result.severity,
        escalate=result.escalate,
    )
