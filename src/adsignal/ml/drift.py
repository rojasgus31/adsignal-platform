"""Feature drift between a training reference window and a current window.

Two complementary signals per feature:

- Population Stability Index, the standard measure in credit and adtech. It
  bins the reference distribution and sums (current% - reference%) *
  ln(current% / reference%). Below 0.1 is stable; 0.1-0.2 is worth watching;
  above 0.2 is a material shift.
- A two-sample Kolmogorov-Smirnov test, which catches shape changes that leave
  the binned proportions roughly intact.

Implemented directly rather than via a monitoring library: it is forty lines,
adds no dependency, and every number can be explained.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

PSI_DRIFT_THRESHOLD = 0.2
KS_SIGNIFICANCE = 0.01

_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """Drift measurement for a single feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Return the PSI between two samples. 0.0 when either side has no data."""
    reference_values = reference.dropna().to_numpy(dtype="float64")
    current_values = current.dropna().to_numpy(dtype="float64")
    if reference_values.size == 0 or current_values.size == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference_values, quantiles))
    if edges.size < 2:
        # A constant reference has no distribution to depart from.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = _share(reference_values, edges)
    current_share = _share(current_values, edges)

    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
) -> list[FeatureDrift]:
    """Measure drift for each column present in both frames.

    Columns absent from either frame are skipped rather than raising: a feature
    set that changed between runs is itself worth reporting, not crashing on.
    """
    reports: list[FeatureDrift] = []
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue

        reference_values = reference[column].dropna().to_numpy(dtype="float64")
        current_values = current[column].dropna().to_numpy(dtype="float64")

        psi = population_stability_index(reference[column], current[column])

        if reference_values.size and current_values.size:
            result = stats.ks_2samp(reference_values, current_values)
            statistic, pvalue = float(result.statistic), float(result.pvalue)
        else:
            statistic, pvalue = 0.0, 1.0

        reports.append(
            FeatureDrift(
                feature=column,
                psi=psi,
                ks_statistic=statistic,
                ks_pvalue=pvalue,
                drifted=psi > PSI_DRIFT_THRESHOLD or pvalue < KS_SIGNIFICANCE,
            )
        )
    return reports


def _share(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of `values` falling in each bin, floored away from zero."""
    counts, _ = np.histogram(values, bins=edges)
    share = counts / max(counts.sum(), 1)
    return cast(np.ndarray, np.clip(share, _EPSILON, None))
