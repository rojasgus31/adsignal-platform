"""The quality gate's policy, exercised without a database or an environment.

That it is testable this way at all is the point of finding 9: the gate used
to read `Settings` and the mart itself, so nothing about its verdict could be
asserted without both.
"""

import pandas as pd

from adsignal.ml.evaluation import (
    MINIMUM_DETECTOR_AUC,
    MINIMUM_MACRO_F1,
    DetectorMetrics,
    PipelineMetrics,
    TriageMetrics,
)
from adsignal.ml.gate import gate_failures, report_lines
from adsignal.ml.training import TrainingResult


def _result(macro_f1: float, roc_auc: float) -> TrainingResult:
    """A `TrainingResult` carrying only the metrics the gate looks at."""
    empty = pd.DataFrame()
    return TrainingResult(
        detector=None,  # type: ignore[arg-type]
        classifier=None,  # type: ignore[arg-type]
        train=empty,
        holdout=empty,
        triage_metrics=TriageMetrics(
            accuracy=0.8,
            macro_f1=macro_f1,
            per_class_f1={"budget_pacing": 0.8},
            confusion={"budget_pacing": {"budget_pacing": 4}},
            n_samples=5,
        ),
        detector_metrics=DetectorMetrics(
            precision=0.6, recall=0.9, f1=0.72, roc_auc=roc_auc, n_samples=100, n_flagged=15
        ),
        pipeline_metrics=PipelineMetrics(
            n_flagged=15,
            n_true_anomalies_flagged=9,
            flagged_precision=0.6,
            cause_accuracy_on_true_positives=0.7,
            n_escalated=4,
            false_alarm_rate_among_escalations=0.25,
        ),
    )


def test_a_run_clearing_both_floors_has_no_failures() -> None:
    assert gate_failures(_result(MINIMUM_MACRO_F1 + 0.1, MINIMUM_DETECTOR_AUC + 0.1)) == []


def test_a_triage_regression_is_reported() -> None:
    failures = gate_failures(_result(MINIMUM_MACRO_F1 - 0.1, MINIMUM_DETECTOR_AUC + 0.1))
    assert len(failures) == 1
    assert "triage macro F1" in failures[0]


def test_a_detector_regression_is_reported_independently() -> None:
    """The detector is a hard pre-filter, so triage macro F1 alone cannot catch it."""
    failures = gate_failures(_result(MINIMUM_MACRO_F1 + 0.1, MINIMUM_DETECTOR_AUC - 0.1))
    assert len(failures) == 1
    assert "detector ROC AUC" in failures[0]


def test_both_floors_are_reported_when_both_are_breached() -> None:
    failures = gate_failures(_result(MINIMUM_MACRO_F1 - 0.1, MINIMUM_DETECTOR_AUC - 0.1))
    assert len(failures) == 2


def test_the_report_covers_triage_the_detector_and_the_end_to_end_run() -> None:
    lines = report_lines(_result(MINIMUM_MACRO_F1 + 0.1, MINIMUM_DETECTOR_AUC + 0.1))
    rendered = "\n".join(lines)
    assert "triage macro F1" in rendered
    assert "detector ROC AUC" in rendered
    assert "detector -> classifier" in rendered
