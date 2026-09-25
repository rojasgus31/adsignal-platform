"""One campaign-day's feature row, read the way `POST /score` reads it.

The case this module exists for is the second test: a campaign's own earliest
day cannot be scored. Every baseline and ratio in `FEATURE_COLUMNS` is a
statistic over the seven days BEFORE the row, so a day without them has NaN
for all of them — and a detector handed NaN ratios still returns a
confident-looking number. The read refuses instead, and says by how much the
history falls short.
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import psycopg
import pytest

from adsignal.ml.features import DEFAULT_BASELINE_WINDOW, FEATURE_COLUMNS, build_features
from adsignal.warehouse.feature_reads import FeatureRow, NoFeatureRow, read_feature_row
from adsignal.warehouse.reader import read_campaign_daily

# Same default and same override as every other DSN-driven test module; see
# `tests/conftest.py`, which `pytest` does not make importable by name.
TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


@pytest.fixture(scope="module")
def campaign_dates(_api_seed: str) -> tuple[str, list[date]]:
    """The seeded campaign and every date it has a mart row for, ascending."""
    with psycopg.connect(TEST_DSN) as conn:
        rows = conn.execute(
            "SELECT event_date FROM fct_campaign_daily WHERE campaign_id = %s ORDER BY event_date",
            (_api_seed,),
        ).fetchall()
    return _api_seed, [row[0] for row in rows]


def test_a_campaign_day_with_history_yields_one_row(campaign_dates) -> None:
    campaign_id, dates = campaign_dates
    outcome = read_feature_row(TEST_DSN, campaign_id, dates[30])

    assert isinstance(outcome, FeatureRow)
    assert len(outcome.frame) == 1
    assert set(FEATURE_COLUMNS) <= set(outcome.frame.columns)
    assert outcome.frame["campaign_id"].iloc[0] == campaign_id
    assert outcome.frame["event_date"].iloc[0].date() == dates[30]


def test_a_campaigns_earliest_day_is_refused_with_the_shortfall(campaign_dates) -> None:
    """By construction it has no prior days, so its baselines are all NaN."""
    campaign_id, dates = campaign_dates
    outcome = read_feature_row(TEST_DSN, campaign_id, dates[0])

    assert isinstance(outcome, NoFeatureRow)
    assert "0 day(s) of history" in outcome.reason
    assert str(DEFAULT_BASELINE_WINDOW) in outcome.reason


def test_the_last_day_before_the_baseline_window_is_still_refused(campaign_dates) -> None:
    """The boundary, not just the extreme: day 6 has 6 prior days and needs 7."""
    campaign_id, dates = campaign_dates
    refused = read_feature_row(TEST_DSN, campaign_id, dates[DEFAULT_BASELINE_WINDOW - 1])
    allowed = read_feature_row(TEST_DSN, campaign_id, dates[DEFAULT_BASELINE_WINDOW])

    assert isinstance(refused, NoFeatureRow)
    assert f"only {DEFAULT_BASELINE_WINDOW - 1} day(s)" in refused.reason
    assert isinstance(allowed, FeatureRow)


def test_an_unknown_campaign_is_refused(campaign_dates) -> None:
    _, dates = campaign_dates
    outcome = read_feature_row(TEST_DSN, "cmp_does_not_exist", dates[30])

    assert isinstance(outcome, NoFeatureRow)
    assert "no mart row" in outcome.reason


def test_a_date_the_campaign_has_no_row_for_is_refused(campaign_dates) -> None:
    """The window read is `event_date <= %s`, so a later date still returns rows.

    Scoring the most recent day before the one asked about would answer a
    different question with a plausible-looking number.
    """
    campaign_id, dates = campaign_dates
    beyond = dates[-1] + timedelta(days=5)
    outcome = read_feature_row(TEST_DSN, campaign_id, beyond)

    assert isinstance(outcome, NoFeatureRow)
    assert "no mart row" in outcome.reason


def test_the_bounded_read_reproduces_the_batch_features_exactly(campaign_dates) -> None:
    """The point of reading only ~29 prior days rather than the whole history.

    If the bounded window changed any value, the on-demand score and the
    nightly stored score would be two different opinions wearing one name.
    """
    campaign_id, dates = campaign_dates
    target = dates[-1]

    batch = build_features(read_campaign_daily(TEST_DSN), baseline_window=DEFAULT_BASELINE_WINDOW)
    expected = batch[
        (batch["campaign_id"] == campaign_id) & (batch["event_date"] == pd.Timestamp(target))
    ]
    outcome = read_feature_row(TEST_DSN, campaign_id, target)

    assert isinstance(outcome, FeatureRow)
    for column in FEATURE_COLUMNS:
        actual_value = outcome.frame[column].iloc[0]
        expected_value = expected[column].iloc[0]
        if isinstance(expected_value, float):
            assert np.isclose(actual_value, expected_value, equal_nan=True), column
        else:
            assert actual_value == expected_value, column


def test_a_database_without_the_mart_is_refused_not_an_error(dropped_tables_client) -> None:
    """A serving database the pipeline has never run against answers, not raises."""
    dsn = dropped_tables_client.app.state.settings.postgres_dsn
    outcome = read_feature_row(dsn, "cmp_0000_000", date(2025, 12, 31))

    assert isinstance(outcome, NoFeatureRow)
