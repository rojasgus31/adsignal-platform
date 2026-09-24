"""Supervised triage of detected anomalies.

Predicts the cause, a severity, and whether a human should be paged.

`HistGradientBoostingClassifier` is the model because it handles NaN natively.
That matters here: rate features are genuinely undefined for a campaign's first
days and for zero-denominator days, and imputing them would be a lie the model
then learns from.

What this demonstrates: `anomaly_cause` is assigned by the synthetic generator
from deterministic effect multipliers, so this model recovers an encoded signal
rather than discovering one. The engineering — feature design, training,
versioning, evaluation, monitoring — is the point, not the science.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from adsignal.generator.models import ANOMALY_CAUSES

ESCALATE_SEVERITY = 4

TRIAGE_FEATURES: tuple[str, ...] = (
    "spend_ratio",
    "impressions_ratio",
    "clicks_ratio",
    "conversions_ratio",
    "ctr_ratio",
    "spend_zscore",
    "ctr_zscore",
    "ctr",
    "cvr",
    "cpa",
    "roas",
    "day_of_week",
)


class TriageClassifier:
    """Predicts an anomaly's cause and severity from its feature row."""

    def __init__(self, random_state: int = 0) -> None:
        # min_samples_leaf defaults to 20, tuned for large datasets. Labelled
        # anomalies are rare by construction, so a training window can easily
        # supply only a few dozen rows; the default would then forbid any
        # split at all and the model would degenerate to predicting the
        # class prior. A small leaf size lets it actually learn from small
        # anomaly counts.
        self._cause_model = HistGradientBoostingClassifier(
            random_state=random_state, max_iter=200, min_samples_leaf=5
        )
        self._severity_model = HistGradientBoostingRegressor(
            random_state=random_state, max_iter=200, min_samples_leaf=5
        )
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "TriageClassifier":
        """Fit on the labelled anomalies within `features`."""
        labelled = self._labelled_rows(features)
        causes = labelled["anomaly_cause"]

        if labelled.empty:
            raise ValueError(
                "no labelled anomalies in the training window; "
                "widen the date range or raise the injection rate"
            )
        distinct = causes.nunique()
        if distinct < 2:
            raise ValueError(
                f"a training set with at least 2 distinct causes is required, found {distinct} "
                f"({sorted(causes.unique())}); a single-class fit always predicts that class"
            )

        matrix = self._matrix(labelled)
        self._cause_model.fit(matrix, causes)
        self._severity_model.fit(matrix, labelled["anomaly_severity"].astype(float))
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return cause, confidence, severity and escalate for every row given."""
        self._require_fitted()
        matrix = self._matrix(features)

        probabilities = self._cause_model.predict_proba(matrix)
        causes = self._cause_model.classes_[probabilities.argmax(axis=1)]
        confidence = probabilities.max(axis=1)

        severity = np.clip(np.rint(self._severity_model.predict(matrix)), 1, 5).astype(int)

        return pd.DataFrame(
            {
                "cause": pd.Series(causes, index=features.index, dtype="object"),
                "confidence": pd.Series(confidence, index=features.index, dtype="float64"),
                "severity": pd.Series(severity, index=features.index, dtype="int64"),
                "escalate": pd.Series(
                    severity >= ESCALATE_SEVERITY, index=features.index, dtype="bool"
                ),
            }
        )

    @staticmethod
    def _labelled_rows(features: pd.DataFrame) -> pd.DataFrame:
        for column in ("is_anomaly", "anomaly_cause", "anomaly_severity"):
            if column not in features.columns:
                raise ValueError(f"features frame is missing the label column: {column}")
        labelled = features[features["is_anomaly"].fillna(False).astype(bool)]
        return labelled[labelled["anomaly_cause"].notna()]

    @staticmethod
    def _matrix(features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in TRIAGE_FEATURES if c not in features.columns]
        if missing:
            raise ValueError(f"features frame is missing columns: {missing}")
        return features.loc[:, list(TRIAGE_FEATURES)]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TriageClassifier is not fitted; call fit() first")


def known_causes() -> tuple[str, ...]:
    """The cause vocabulary, for callers validating predictions."""
    return ANOMALY_CAUSES
