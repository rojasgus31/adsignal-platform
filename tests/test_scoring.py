"""Scoring is pure model work: a feature row in, a verdict out.

The models are frame-in, not mapping-in — `AnomalyDetector.predict/score` and
`TriageClassifier.predict` all take a `pandas.DataFrame` carrying the columns
`FEATURE_COLUMNS` declares — so these tests score one-row frames built by
`build_features`, which is the same shape the batch path scores in bulk.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

from adsignal.ml.features import DEFAULT_BASELINE_WINDOW, build_features
from adsignal.ml.scoring import ScoringModels


@pytest.fixture
def feature_frame(labelled_series) -> pd.DataFrame:
    """Features over a 180-day labelled series, the same shape training saw."""
    return build_features(labelled_series(180), baseline_window=DEFAULT_BASELINE_WINDOW)


def _row(frame: pd.DataFrame, position: int) -> pd.DataFrame:
    """One row of `frame`, still a frame — `.iloc[n]` would give a Series."""
    return frame.iloc[[position]]


def test_it_scores_a_feature_row(trained_artifacts, feature_frame) -> None:
    models = ScoringModels.load(trained_artifacts)
    result = models.score(_row(feature_frame, 100))
    assert isinstance(result.anomaly_score, float)
    assert isinstance(result.is_anomaly, bool)


def test_a_flagged_row_gets_a_cause(trained_artifacts, feature_frame) -> None:
    """Triage only runs on rows the detector flags; an unflagged row has no cause."""
    models = ScoringModels.load(trained_artifacts)
    verdicts = [models.score(_row(feature_frame, index)) for index in range(20, 120)]

    flagged = [verdict for verdict in verdicts if verdict.is_anomaly]
    assert flagged, "the detector flagged nothing across 100 days of an injected series"
    for verdict in flagged:
        assert verdict.cause is not None
        assert verdict.confidence is not None
        assert 0.0 <= verdict.confidence <= 1.0
        assert verdict.severity in (1, 2, 3, 4, 5)
    for verdict in verdicts:
        if not verdict.is_anomaly:
            assert verdict.cause is None
            assert verdict.confidence is None


def test_a_flagged_row_scores_higher_than_an_unflagged_one(
    trained_artifacts, feature_frame
) -> None:
    """`is_anomaly` and `anomaly_score` must be two views of one decision.

    Reporting a flag alongside a score computed some other way would let the
    two disagree on the same row, which is worse than reporting neither.
    """
    models = ScoringModels.load(trained_artifacts)
    verdicts = [models.score(_row(feature_frame, index)) for index in range(20, 120)]
    flagged = [v.anomaly_score for v in verdicts if v.is_anomaly]
    unflagged = [v.anomaly_score for v in verdicts if not v.is_anomaly]
    assert flagged and unflagged
    assert min(flagged) > max(unflagged)


def test_a_missing_feature_raises_rather_than_guessing(trained_artifacts, feature_frame) -> None:
    """Silently imputing a missing feature produces a confident wrong answer."""
    models = ScoringModels.load(trained_artifacts)
    crippled = _row(feature_frame, 100).drop(columns=["ctr_ratio"])
    with pytest.raises(ValueError, match="ctr_ratio"):
        models.score(crippled)


def test_scoring_many_rows_at_once_is_refused(trained_artifacts, feature_frame) -> None:
    """A verdict describes one campaign-day; silently scoring the first of many
    would answer a question the caller did not ask."""
    models = ScoringModels.load(trained_artifacts)
    with pytest.raises(ValueError, match="exactly one feature row"):
        models.score(feature_frame.iloc[10:14])


def test_a_directory_without_artifacts_raises(tmp_path) -> None:
    """A missing artifact is a broken run, not a reason to score with one model."""
    with pytest.raises(FileNotFoundError):
        ScoringModels.load(tmp_path)


def test_scoring_layer_imports_nothing_forbidden() -> None:
    """`ml/` must not reach into warehouse, pipeline, config or api."""
    source = Path(__file__).resolve().parents[1] / "src/adsignal/ml/scoring.py"
    for node in ast.walk(ast.parse(source.read_text())):
        name = ""
        if isinstance(node, ast.Import):
            name = node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
        assert not any(
            name.startswith(f"adsignal.{bad}") for bad in ("warehouse", "pipeline", "config", "api")
        ), f"ml/scoring.py imports {name}"


def test_loading_the_scoring_layer_does_not_import_mlflow() -> None:
    """MLflow costs ~136 MB resident and the API never uses it.

    `ml/scoring.py` reaches `load_models` through `ml/registry.py`, which is
    also the module that logs training runs to MLflow. A module-scope
    `import mlflow` there would put all of it inside a 512 MB API container to
    unpickle two files — see `scripts/measure_memory.py`.
    """
    import subprocess
    import sys

    probe = "import sys, adsignal.ml.scoring; sys.exit(1 if 'mlflow' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0
