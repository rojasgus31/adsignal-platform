"""`GET /operations`."""

from fastapi import APIRouter, Depends, Query

from adsignal.api.deps import get_settings
from adsignal.api.schemas import RunOut
from adsignal.config import Settings
from adsignal.warehouse.run_log import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, RunEntry, read_runs

router = APIRouter(tags=["operations"])


def _to_run_out(entry: RunEntry) -> RunOut:
    return RunOut(
        run_id=entry.run_id,
        asset=entry.asset,
        started_at=entry.started_at,
        duration_seconds=entry.duration_seconds,
        rows_written=entry.rows_written,
        status=entry.status,
        detail=entry.detail,
    )


@router.get("/operations", response_model=list[RunOut])
def get_operations(
    settings: Settings = Depends(get_settings),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> list[RunOut]:
    """List one page of the pipeline's own operations log, newest first.

    Bounded, like `/models`: the log grows with every asset materialization.
    Serves what survives an ephemeral CI runner after the run that wrote it
    is gone.
    """
    runs = read_runs(settings.postgres_dsn, limit=limit, offset=offset)
    return [_to_run_out(run) for run in runs]
