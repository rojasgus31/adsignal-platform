"""The one training recipe: build features, split, fit both models, evaluate.

`adsignal.pipeline.assets.trained_models` (the Dagster asset) and
`adsignal.ml.gate` (the CI entry point) both need to do exactly this - and
before this module existed, they each did it independently, in eleven
near-identical lines. That let them silently diverge: `assets.py` passed
`baseline_window=7` explicitly while `gate.py` relied on `build_features`'s
default, so they agreed only by coincidence, and a future change to one
without the other would mean the CI gate certifies a model different from
the one the pipeline actually trains. This is the single place the recipe
lives now; the asset and the CLI are two adapters over one use case.
"""

from dataclasses import dataclass

import pandas as pd

from adsignal.ml.detector import AnomalyDetector
from adsignal.ml.evaluation import (
    DetectorMetrics,
    PipelineMetrics,
    TriageMetrics,
    evaluate_detector,
    evaluate_pipeline,
    evaluate_triage,
)
from adsignal.ml.features import build_features
from adsignal.ml.splits import temporal_split
from adsignal.ml.triage import TriageClassifier

HOLDOUT_FRACTION = 0.25
BASELINE_WINDOW = 7


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Everything one training run produces: both models, their split, and their metrics."""

    detector: AnomalyDetector
    classifier: TriageClassifier
    train: pd.DataFrame
    holdout: pd.DataFrame
    triage_metrics: TriageMetrics
    detector_metrics: DetectorMetrics
    pipeline_metrics: PipelineMetrics


def train_and_evaluate(mart: pd.DataFrame, *, seed: int) -> TrainingResult:
    """Build features from `mart`, split temporally, fit both models, and evaluate."""
    features = build_features(mart, baseline_window=BASELINE_WINDOW)
    train, holdout = temporal_split(features, holdout_fraction=HOLDOUT_FRACTION)

    detector = AnomalyDetector(random_state=seed).fit(train)
    classifier = TriageClassifier(random_state=seed).fit(train)

    triage_metrics = evaluate_triage(classifier, holdout)
    detector_metrics = evaluate_detector(detector, holdout)
    pipeline_metrics = evaluate_pipeline(detector, classifier, holdout)

    return TrainingResult(
        detector=detector,
        classifier=classifier,
        train=train,
        holdout=holdout,
        triage_metrics=triage_metrics,
        detector_metrics=detector_metrics,
        pipeline_metrics=pipeline_metrics,
    )
