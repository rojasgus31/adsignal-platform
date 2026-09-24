"""Unsupervised anomaly detection over campaign-day features.

IsolationForest on the ratio and z-score features — the ones that express "this
day departs from this campaign's own recent behaviour". Raw magnitudes are
deliberately excluded: a campaign that always spends 5000 is not anomalous for
spending 5000.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DETECTOR_FEATURES: tuple[str, ...] = (
    "spend_ratio",
    "impressions_ratio",
    "clicks_ratio",
    "conversions_ratio",
    "ctr_ratio",
    "spend_zscore",
    "ctr_zscore",
)


class AnomalyDetector:
    """IsolationForest wrapper with the project's NaN policy baked in.

    Ratio features are NaN for a campaign's first `baseline_window` days. Those
    rows carry no evidence either way, so `_matrix` centres them at the neutral
    value (1.0 for ratios, 0.0 for z-scores) and subtracts that neutral, so a
    missing ratio contributes zero deviation. The feature frame itself keeps
    its NaNs — only the model boundary substitutes.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 0) -> None:
        self._forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        """Fit on the detector's feature subset. Returns self for chaining."""
        matrix = self._matrix(features)
        if matrix.shape[0] == 0:
            raise ValueError("cannot fit an anomaly detector on no rows")
        self._forest.fit(matrix)
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return a boolean array: True where the row is anomalous."""
        self._require_fitted()
        raw = self._forest.predict(self._matrix(features))
        return np.asarray(raw == -1)

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores. Higher means more anomalous."""
        self._require_fitted()
        return np.asarray(-self._forest.score_samples(self._matrix(features)))

    def _matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in DETECTOR_FEATURES if c not in features.columns]
        if missing:
            raise ValueError(f"features frame is missing columns: {missing}")
        centred = features.loc[:, list(DETECTOR_FEATURES)].copy()
        for column in centred.columns:
            neutral = 1.0 if column.endswith("_ratio") else 0.0
            centred[column] = centred[column].fillna(neutral) - neutral
        return centred

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("AnomalyDetector is not fitted; call fit() first")
