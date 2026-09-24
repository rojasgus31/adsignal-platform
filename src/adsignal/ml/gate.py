"""CI entry point: train, evaluate, and fail the build on a quality regression.

Run as `python -m adsignal.ml.gate`. Exits non-zero when triage macro F1 falls
below the committed floor, or when the detector's ROC AUC falls below its own
floor, so a model regression cannot reach main unnoticed.

The detector is gated separately from the classifier because it is a hard
pre-filter: the scoring asset only triages the rows the detector flags. A
detector whose signal has degraded silently starves the classifier of the
anomalies it should see while triage metrics still look fine, so triage
macro F1 alone cannot catch that failure mode.

The detector is gated on ROC AUC rather than recall: recall is threshold- and
base-rate-dependent (the same detector's recall moves with the dataset's
anomaly rate, since contamination is derived from it), so a fixed recall
floor is not a stable regression gate. AUC asks a base-rate-independent
question instead - does the detector's score still separate anomalies from
normal days at all - which is what a regression gate should ask. See
`adsignal.ml.evaluation.MINIMUM_DETECTOR_AUC` for the full rationale.
"""

import sys

from adsignal.config import get_settings
from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.evaluation import (
    MINIMUM_DETECTOR_AUC,
    MINIMUM_MACRO_F1,
    evaluate_detector,
    evaluate_triage,
)
from adsignal.ml.features import build_features
from adsignal.ml.splits import temporal_split
from adsignal.ml.triage import TriageClassifier
from adsignal.warehouse.reader import read_campaign_daily


def main() -> int:
    """Train on the mart, evaluate on the holdout, and gate on both models."""
    settings = get_settings()
    features = build_features(read_campaign_daily(settings.postgres_dsn))
    train, holdout = temporal_split(features, holdout_fraction=0.25)

    detector = AnomalyDetector(random_state=settings.seed).fit(train)
    classifier = TriageClassifier(random_state=settings.seed).fit(train)

    triage_metrics = evaluate_triage(classifier, holdout)
    detector_metrics = evaluate_detector(detector, holdout)

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
