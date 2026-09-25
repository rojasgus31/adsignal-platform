"""The model quality gate: does a training run clear both committed floors?

Pure policy over a `TrainingResult`. It reads no configuration and touches no
database — `ml/` must not import `config`, `warehouse`, `pipeline` or `api`,
and this module previously broke that rule twice, which made the quality gate
untestable without an environment and a populated mart. The composition root
that supplies the mart and turns this into an exit code lives in
`adsignal.cli.quality_gate` (`python -m adsignal.cli.quality_gate`).

The gate fails when triage macro F1 falls below the committed floor, or when
the detector's ROC AUC falls below its own floor, so a model regression
cannot reach main unnoticed.

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

from adsignal.ml.evaluation import (
    MINIMUM_DETECTOR_AUC,
    MINIMUM_MACRO_F1,
    DetectorMetrics,
    PipelineMetrics,
    TriageMetrics,
)
from adsignal.ml.training import TrainingResult


def report_lines(result: TrainingResult) -> list[str]:
    """Render the full human-readable report for one training run."""
    return [
        *_triage_lines(result.triage_metrics),
        *_detector_lines(result.detector_metrics),
        "",
        "--- end-to-end (detector -> classifier, as production runs it) ---",
        *_pipeline_lines(result.pipeline_metrics),
    ]


def gate_failures(result: TrainingResult) -> list[str]:
    """Return one line per breached floor. Empty means the run passes."""
    failures: list[str] = []
    triage = result.triage_metrics
    detector = result.detector_metrics
    if not triage.meets_gate():
        failures.append(
            f"FAIL: triage macro F1 {triage.macro_f1:.4f} is below the floor {MINIMUM_MACRO_F1}"
        )
    if not detector.meets_gate():
        failures.append(
            f"FAIL: detector ROC AUC {detector.roc_auc:.4f} "
            f"is below the floor {MINIMUM_DETECTOR_AUC}"
        )
    return failures


def _triage_lines(triage: TriageMetrics) -> list[str]:
    return [
        f"triage macro F1: {triage.macro_f1:.4f} (floor {MINIMUM_MACRO_F1})",
        f"triage accuracy: {triage.accuracy:.4f} over {triage.n_samples} samples",
        *(f"  {cause}: {score:.4f}" for cause, score in sorted(triage.per_class_f1.items())),
    ]


def _detector_lines(detector: DetectorMetrics) -> list[str]:
    return [
        f"detector ROC AUC: {detector.roc_auc:.4f} (floor {MINIMUM_DETECTOR_AUC})",
        f"detector precision: {detector.precision:.4f}, "
        f"recall: {detector.recall:.4f}, "
        f"F1: {detector.f1:.4f}, "
        f"flagged {detector.n_flagged} of {detector.n_samples} samples",
    ]


def _pipeline_lines(pipeline: PipelineMetrics) -> list[str]:
    return [
        f"flagged {pipeline.n_flagged} rows, "
        f"{pipeline.n_true_anomalies_flagged} of them true anomalies "
        f"(flagged precision {pipeline.flagged_precision:.4f})",
        f"cause accuracy on flagged true anomalies: "
        f"{pipeline.cause_accuracy_on_true_positives:.4f}",
        f"escalated {pipeline.n_escalated} rows, "
        f"false-alarm rate among them: {pipeline.false_alarm_rate_among_escalations:.4f}",
    ]
