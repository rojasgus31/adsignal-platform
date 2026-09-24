"""CI entry point: train, evaluate, and fail the build on a quality regression.

Run as `python -m adsignal.ml.gate`. Exits non-zero when triage macro F1 falls
below the committed floor, or when the detector's ROC AUC falls below its own
floor, so a model regression cannot reach main unnoticed.

The detector is gated separately from the classifier because it is a hard
pre-filter: the scoring asset only triages the rows the detector flags. A
detector whose signal has degraded silently starves the classifier of the
anomalies it should see while triage metrics still look fine, so triage
macro F1 alone cannot catch that failure mode.

The detector's contamination is derived from the training window's observed
anomaly rate (see `AnomalyDetector._resolve_contamination`), which a sweep on
real pipeline data confirmed is the F1-optimal operating point - not a
recall-oriented one. At a ~5% base rate that costs about two thirds of
recall (0.31 measured): most real anomalies never reach the classifier at
all. That is the trade accepted, because a lower-precision pre-filter makes
the downstream triage output unusable. It is also why recall cannot be the
regression gate: recall is threshold- and base-rate-dependent (the same
detector's recall moves with the dataset's anomaly rate, since contamination
is derived from it), so a fixed recall floor is not stable. AUC asks a
base-rate-independent question instead - does the detector's score still
separate anomalies from normal days at all - which is what a regression gate
should ask. See `adsignal.ml.evaluation.MINIMUM_DETECTOR_AUC` for the full
rationale.

The end-to-end section below reports, but does not gate on, what the
pipeline looks like run the way production actually runs it: the classifier
scoring whatever the detector flagged, true anomaly or not - not the
holdout's true anomalies directly, which is what `evaluate_triage` alone
would show. See `adsignal.ml.evaluation.evaluate_pipeline`.
"""

import sys

from adsignal.config import get_settings
from adsignal.ml.evaluation import MINIMUM_DETECTOR_AUC, MINIMUM_MACRO_F1
from adsignal.ml.training import train_and_evaluate
from adsignal.warehouse.reader import read_campaign_daily


def main() -> int:
    """Train on the mart, evaluate on the holdout, and gate on both models."""
    settings = get_settings()
    result = train_and_evaluate(read_campaign_daily(settings.postgres_dsn), seed=settings.seed)
    triage_metrics = result.triage_metrics
    detector_metrics = result.detector_metrics
    pipeline_metrics = result.pipeline_metrics

    print(f"triage macro F1: {triage_metrics.macro_f1:.4f} (floor {MINIMUM_MACRO_F1})")
    print(f"triage accuracy: {triage_metrics.accuracy:.4f} over {triage_metrics.n_samples} samples")
    for cause, score in sorted(triage_metrics.per_class_f1.items()):
        print(f"  {cause}: {score:.4f}")

    print(f"detector ROC AUC: {detector_metrics.roc_auc:.4f} (floor {MINIMUM_DETECTOR_AUC})")
    print(
        f"detector precision: {detector_metrics.precision:.4f}, "
        f"recall: {detector_metrics.recall:.4f}, "
        f"F1: {detector_metrics.f1:.4f}, "
        f"flagged {detector_metrics.n_flagged} of {detector_metrics.n_samples} samples"
    )

    print("\n--- end-to-end (detector -> classifier, as production runs it) ---")
    print(
        f"flagged {pipeline_metrics.n_flagged} rows, "
        f"{pipeline_metrics.n_true_anomalies_flagged} of them true anomalies "
        f"(flagged precision {pipeline_metrics.flagged_precision:.4f})"
    )
    print(
        f"cause accuracy on flagged true anomalies: "
        f"{pipeline_metrics.cause_accuracy_on_true_positives:.4f}"
    )
    print(
        f"escalated {pipeline_metrics.n_escalated} rows, "
        f"false-alarm rate among them: {pipeline_metrics.false_alarm_rate_among_escalations:.4f}"
    )

    triage_ok = triage_metrics.meets_gate()
    detector_ok = detector_metrics.meets_gate()

    if not triage_ok:
        print(
            f"FAIL: triage macro F1 {triage_metrics.macro_f1:.4f} "
            f"is below the floor {MINIMUM_MACRO_F1}"
        )
    if not detector_ok:
        print(
            f"FAIL: detector ROC AUC {detector_metrics.roc_auc:.4f} "
            f"is below the floor {MINIMUM_DETECTOR_AUC}"
        )

    if not (triage_ok and detector_ok):
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
