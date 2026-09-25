"""`GET /models/{run_id}/evidence`."""

from fastapi import APIRouter, Depends, HTTPException

from adsignal.api.deps import get_settings
from adsignal.api.schemas import EvidenceOut
from adsignal.config import Settings
from adsignal.warehouse.serving_reads import ModelEvidence, read_model_evidence

router = APIRouter(tags=["models"])


def _to_evidence_out(evidence: ModelEvidence) -> EvidenceOut:
    return EvidenceOut(
        run_id=evidence.run_id,
        trained_at=evidence.trained_at,
        holdout_samples=evidence.holdout_samples,
        per_class_f1=evidence.per_class_f1,
        confusion=evidence.confusion,
        detector_roc_auc=evidence.detector_roc_auc,
        detector_f1=evidence.detector_f1,
        triage_macro_f1=evidence.triage_macro_f1,
    )


@router.get("/models/{run_id}/evidence", response_model=EvidenceOut)
def get_evidence(run_id: str, settings: Settings = Depends(get_settings)) -> EvidenceOut:
    """Everything stored about one training run's performance.

    The confusion matrix and per-class F1 scores have been sitting in
    `model_metrics` since the ML layer was built; this is the first reader.
    """
    evidence = read_model_evidence(settings.postgres_dsn, run_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail=f"no training run {run_id!r}")
    return _to_evidence_out(evidence)
