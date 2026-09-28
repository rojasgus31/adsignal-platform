"""Pydantic response and request models for the API.

These are the API's own DTOs, distinct from the warehouse's dataclasses
(`SearchHit`, `StoredNarrative`, ...) and the pandas frames `reader.py`
returns — the routers translate between the two.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


class CampaignSummary(BaseModel):
    """One row of `GET /campaigns`.

    Both name fields are always present and never null. When `dim_campaign`
    has no row for the campaign, or the row it has carries a blank name, they
    carry the corresponding identifier - so a dashboard rendering
    `campaign_name` shows an id rather than an empty cell. See
    `warehouse.campaign_reads._LIST_CAMPAIGNS_SQL`, which is where both the
    fallback and the blank-name handling live.
    """

    campaign_id: str
    campaign_name: str
    advertiser_id: str
    advertiser_name: str
    channel: str


class MetricPoint(BaseModel):
    """One day of a campaign's metric time series."""

    event_date: date
    impressions: int
    clicks: int
    conversions: int
    spend: float
    revenue: float
    is_anomaly: bool


class CampaignMetrics(BaseModel):
    """The full time series for one campaign, as returned by `GET /campaigns/{id}/metrics`."""

    campaign_id: str
    points: list[MetricPoint]


class AnomalyOut(BaseModel):
    """One row of `GET /anomalies` and of `GET /anomalies/{campaign_id}/{event_date}`.

    `campaign_name` is always present and never null; it carries
    `campaign_id` when `dim_campaign` has no row for the campaign or that
    row's name is blank. It is joined onto the prediction in the warehouse
    read, so labelling this table with campaign names costs the client no
    extra request.
    """

    campaign_id: str
    campaign_name: str
    event_date: date
    cause: str
    confidence: float
    severity: int
    escalate: bool
    anomaly_score: float


class ModelMetricsOut(BaseModel):
    """One row of `GET /models`: a single training run's evaluation."""

    run_id: str
    trained_at: datetime
    triage_accuracy: float
    triage_macro_f1: float
    detector_precision: float
    detector_recall: float
    detector_f1: float
    detector_roc_auc: float


class EvidenceOut(BaseModel):
    """`GET /models/{run_id}/evidence`: everything stored about one training run."""

    run_id: str
    trained_at: datetime
    holdout_samples: int
    per_class_f1: dict[str, float]
    confusion: dict[str, dict[str, int]]
    detector_roc_auc: float
    detector_f1: float
    triage_macro_f1: float


class BenchmarkRun(BaseModel):
    """One run's summary for the dashboard's comparison view."""

    run_id: str
    trained_at: datetime
    triage_macro_f1: float
    detector_roc_auc: float


class BenchmarkOut(BaseModel):
    """`GET /benchmark`: every run, plus which narrative provider answered most recently."""

    runs: list[BenchmarkRun]
    latest_narrative_provider: str | None = None


class RunOut(BaseModel):
    """One row of `GET /operations`: one asset's execution within one pipeline run."""

    run_id: str
    asset: str
    started_at: datetime
    duration_seconds: float
    rows_written: int
    status: str
    detail: str


class SearchRequest(BaseModel):
    """`POST /search` body.

    `limit` is bounded: an unbounded limit on a public demo is a
    denial-of-service lever.
    """

    query: str = Field(min_length=1)
    limit: int = Field(10, ge=1, le=50)
    kind: str | None = None

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        """A whitespace-only query is not a valid search, so it is a 422, not a
        zero-hit 200 or an embedding-side `ValueError`."""
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchHitOut(BaseModel):
    """One nearest-neighbour hit."""

    entity_kind: str
    entity_id: str
    text: str
    distance: float


class SearchResponse(BaseModel):
    """`POST /search` response."""

    hits: list[SearchHitOut]


class ScoreRequest(BaseModel):
    """`POST /score` body: the one campaign-day to score."""

    campaign_id: str = Field(min_length=1)
    event_date: date

    @field_validator("campaign_id")
    @classmethod
    def _reject_blank_campaign_id(cls, value: str) -> str:
        """A whitespace-only id is a malformed request, not a 404."""
        if not value.strip():
            raise ValueError("campaign_id must not be blank")
        return value


class ScoreOut(BaseModel):
    """`POST /score` response: one campaign-day's verdict, computed on demand.

    The campaign-day is echoed back so a caller can tell this apart from a
    cached or stored answer about some other day.

    `cause`, `confidence`, `severity` and `escalate` are null unless the
    detector flagged the row: the triage classifier is fitted only on
    labelled anomalies, so its opinion about an unremarkable day is its class
    prior, not a finding.
    """

    campaign_id: str
    event_date: date
    anomaly_score: float
    is_anomaly: bool
    cause: str | None = None
    confidence: float | None = None
    severity: int | None = None
    escalate: bool | None = None
