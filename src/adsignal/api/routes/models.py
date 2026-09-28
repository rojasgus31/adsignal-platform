"""`GET /models`, `GET /benchmark`."""

from fastapi import APIRouter, Depends, Query

from adsignal.api.deps import get_settings
from adsignal.api.schemas import BenchmarkOut, BenchmarkRun, ModelMetricsOut
from adsignal.config import Settings
from adsignal.warehouse.narrative_store import read_latest_narrative
from adsignal.warehouse.paging import DEFAULT_SERVING_PAGE_LIMIT, MAX_PAGE_LIMIT
from adsignal.warehouse.serving_reads import ModelMetricsRow, read_model_metrics

router = APIRouter(tags=["models"])


def _to_model_metrics_out(row: ModelMetricsRow) -> ModelMetricsOut:
    return ModelMetricsOut(
        run_id=row.run_id,
        trained_at=row.trained_at,
        triage_accuracy=row.triage_accuracy,
        triage_macro_f1=row.triage_macro_f1,
        detector_precision=row.detector_precision,
        detector_recall=row.detector_recall,
        detector_f1=row.detector_f1,
        detector_roc_auc=row.detector_roc_auc,
    )


@router.get("/models", response_model=list[ModelMetricsOut])
def get_models(
    settings: Settings = Depends(get_settings),
    limit: int = Query(DEFAULT_SERVING_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> list[ModelMetricsOut]:
    """List one page of training-run evaluations, newest first.

    Bounded, like `/search` and `/ask`: the history grows with every nightly
    run. Empty before the first run.
    """
    rows = read_model_metrics(settings.postgres_dsn, limit=limit, offset=offset)
    return [_to_model_metrics_out(row) for row in rows]


@router.get("/benchmark", response_model=BenchmarkOut)
def get_benchmark(settings: Settings = Depends(get_settings)) -> BenchmarkOut:
    """The dashboard's comparison view: recent runs, plus who narrated most recently.

    Reads through the same bounded reader as `/models`, so one response can
    never grow with the training history either.
    """
    runs = [
        BenchmarkRun(
            run_id=row.run_id,
            trained_at=row.trained_at,
            triage_macro_f1=row.triage_macro_f1,
            detector_roc_auc=row.detector_roc_auc,
        )
        for row in read_model_metrics(settings.postgres_dsn)
    ]
    narrative = read_latest_narrative(settings.postgres_dsn)
    return BenchmarkOut(
        runs=runs,
        latest_narrative_provider=narrative.provider if narrative is not None else None,
    )
