import numpy as np
import pandas as pd

from adsignal.ml.drift import (
    PSI_DRIFT_THRESHOLD,
    feature_drift,
    population_stability_index,
)


def _normal(loc: float, size: int = 2000, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(loc, 1.0, size))


def test_psi_is_near_zero_for_identical_distributions() -> None:
    values = _normal(0.0)
    assert population_stability_index(values, values) < 0.01


def test_psi_grows_when_the_distribution_shifts() -> None:
    reference = _normal(0.0, seed=1)
    mild = _normal(0.3, seed=2)
    severe = _normal(2.0, seed=3)
    assert population_stability_index(reference, mild) < population_stability_index(
        reference, severe
    )


def test_psi_crosses_the_threshold_for_a_large_shift() -> None:
    reference = _normal(0.0, seed=1)
    severe = _normal(2.0, seed=2)
    assert population_stability_index(reference, severe) > PSI_DRIFT_THRESHOLD


def test_psi_handles_all_nan_input_without_crashing() -> None:
    empty = pd.Series([np.nan] * 100)
    assert population_stability_index(_normal(0.0), empty) == 0.0


def test_psi_handles_a_constant_column() -> None:
    constant = pd.Series([5.0] * 500)
    assert population_stability_index(constant, constant) == 0.0


def test_feature_drift_reports_one_row_per_column() -> None:
    reference = pd.DataFrame({"a": _normal(0.0, seed=1), "b": _normal(0.0, seed=2)})
    current = pd.DataFrame({"a": _normal(2.0, seed=3), "b": _normal(0.0, seed=4)})
    reports = feature_drift(reference, current, ["a", "b"])
    assert [report.feature for report in reports] == ["a", "b"]


def test_feature_drift_flags_only_the_shifted_column() -> None:
    reference = pd.DataFrame({"a": _normal(0.0, seed=1), "b": _normal(0.0, seed=2)})
    current = pd.DataFrame({"a": _normal(3.0, seed=3), "b": _normal(0.0, seed=4)})
    by_name = {report.feature: report for report in feature_drift(reference, current, ["a", "b"])}
    assert by_name["a"].drifted
    assert not by_name["b"].drifted


def test_ks_pvalue_is_a_probability() -> None:
    reference = pd.DataFrame({"a": _normal(0.0, seed=1)})
    current = pd.DataFrame({"a": _normal(1.0, seed=2)})
    report = feature_drift(reference, current, ["a"])[0]
    assert 0.0 <= report.ks_pvalue <= 1.0
    assert 0.0 <= report.ks_statistic <= 1.0


def test_missing_column_is_skipped_rather_than_raising() -> None:
    reference = pd.DataFrame({"a": _normal(0.0)})
    current = pd.DataFrame({"a": _normal(0.0)})
    reports = feature_drift(reference, current, ["a", "does_not_exist"])
    assert [report.feature for report in reports] == ["a"]
