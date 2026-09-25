"""`GET /anomalies`, `GET /anomalies/{campaign_id}/{event_date}`."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from adsignal.api.deps import get_settings
from adsignal.api.schemas import AnomalyOut
from adsignal.config import Settings
from adsignal.warehouse.serving_reads import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    AnomalyPredictionRow,
    read_anomaly_prediction,
    read_anomaly_predictions,
)

router = APIRouter(tags=["anomalies"])


def _to_anomaly_out(row: AnomalyPredictionRow) -> AnomalyOut:
    return AnomalyOut(
        campaign_id=row.campaign_id,
        event_date=row.event_date,
        cause=row.cause,
        confidence=row.confidence,
        severity=row.severity,
        escalate=row.escalate,
        anomaly_score=row.anomaly_score,
    )


@router.get("/anomalies", response_model=list[AnomalyOut])
def get_anomalies(
    settings: Settings = Depends(get_settings),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> list[AnomalyOut]:
    """List one page of current anomaly predictions, most recent day first.

    Bounded, like `/search` and `/ask`: every campaign-day the current model
    flags is a row here, so an unbounded response grows with the table. Empty
    before the first pipeline run.
    """
    rows = read_anomaly_predictions(settings.postgres_dsn, limit=limit, offset=offset)
    return [_to_anomaly_out(row) for row in rows]


@router.get("/anomalies/{campaign_id}/{event_date}", response_model=AnomalyOut)
def get_anomaly(
    campaign_id: str, event_date: date, settings: Settings = Depends(get_settings)
) -> AnomalyOut:
    """Return one campaign-day's prediction, or 404 if there is none."""
    row = read_anomaly_prediction(settings.postgres_dsn, campaign_id, event_date)
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"no prediction for {campaign_id} on {event_date}"
        )
    return _to_anomaly_out(row)
