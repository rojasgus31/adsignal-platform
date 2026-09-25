"""`POST /score` — the endpoint that runs a model instead of reading a row.

What these assert, in order of what matters: that a scoreable campaign-day
comes back with a verdict computed now; that a campaign-day which cannot be
scored is refused with the reason rather than given a number; that the
refusal distinguishes "this day has no features" (404) from "this process has
no models" (503); and that the endpoint is throttled like the other one that
runs live inference.
"""

import os
from datetime import date, timedelta

import psycopg
import pytest

from adsignal.ml.features import DEFAULT_BASELINE_WINDOW
from adsignal.ml.triage import known_causes

TEST_DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


@pytest.fixture(scope="module")
def scoreable_day(_api_seed: str) -> tuple[str, date]:
    """A seeded campaign and a day of it with more than enough prior history."""
    with psycopg.connect(TEST_DSN) as conn:
        rows = conn.execute(
            "SELECT event_date FROM fct_campaign_daily WHERE campaign_id = %s ORDER BY event_date",
            (_api_seed,),
        ).fetchall()
    return _api_seed, rows[-1][0]


@pytest.fixture(scope="module")
def earliest_day(_api_seed: str) -> tuple[str, date]:
    """A seeded campaign and its own first day, which has no prior history at all."""
    with psycopg.connect(TEST_DSN) as conn:
        row = conn.execute(
            "SELECT MIN(event_date) FROM fct_campaign_daily WHERE campaign_id = %s",
            (_api_seed,),
        ).fetchone()
    assert row is not None
    return _api_seed, row[0]


def _score(client, campaign_id: str, event_date: date | str):
    day = event_date.isoformat() if isinstance(event_date, date) else event_date
    return client.post("/score", json={"campaign_id": campaign_id, "event_date": day})


def test_scoring_a_real_campaign_day_returns_a_verdict(client, scoreable_day) -> None:
    campaign_id, event_date = scoreable_day
    response = _score(client, campaign_id, event_date)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["campaign_id"] == campaign_id
    assert payload["event_date"] == event_date.isoformat()
    assert isinstance(payload["anomaly_score"], float)
    assert isinstance(payload["is_anomaly"], bool)


def test_an_unflagged_day_carries_no_cause_and_a_flagged_one_does(client, scoreable_day) -> None:
    """Triage is fitted only on labelled anomalies; its view of a quiet day is a prior."""
    campaign_id, event_date = scoreable_day
    payload = _score(client, campaign_id, event_date).json()

    if payload["is_anomaly"]:
        assert payload["cause"] in known_causes()
        assert 0.0 <= payload["confidence"] <= 1.0
        assert payload["severity"] in (1, 2, 3, 4, 5)
        assert isinstance(payload["escalate"], bool)
    else:
        assert payload["cause"] is None
        assert payload["confidence"] is None
        assert payload["severity"] is None


def test_a_campaigns_earliest_day_is_404_with_the_shortfall(client, earliest_day) -> None:
    """The failure the whole endpoint is built around.

    That day's rolling baselines are NaN, and a detector handed NaN ratios
    still returns a confident number. A confidently wrong score is worse than
    a refusal, so this is a 404 that names how short the history is.
    """
    campaign_id, event_date = earliest_day
    response = _score(client, campaign_id, event_date)

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "history" in detail
    assert str(DEFAULT_BASELINE_WINDOW) in detail


def test_an_unknown_campaign_day_is_404_not_500(client, scoreable_day) -> None:
    """A confidently wrong score is worse than a refusal."""
    _, event_date = scoreable_day
    response = _score(client, "cmp_does_not_exist", event_date)

    assert response.status_code == 404
    assert "no mart row" in response.json()["detail"]


def test_a_day_the_campaign_has_no_row_for_is_404(client, scoreable_day) -> None:
    campaign_id, event_date = scoreable_day
    response = _score(client, campaign_id, event_date + timedelta(days=30))

    assert response.status_code == 404


def test_a_malformed_date_is_422(client) -> None:
    response = _score(client, "cmp_0000_000", "not-a-date")
    assert response.status_code == 422


def test_a_blank_campaign_id_is_422(client, scoreable_day) -> None:
    _, event_date = scoreable_day
    assert _score(client, "   ", event_date).status_code == 422


def test_score_is_rate_limited(client, scoreable_day) -> None:
    """The only other endpoint doing live model work is throttled; so is this one."""
    campaign_id, event_date = scoreable_day
    statuses = [_score(client, campaign_id, event_date).status_code for _ in range(40)]

    assert 429 in statuses
    assert statuses[0] != 429


def test_score_on_a_fresh_database_is_503_not_500(empty_client) -> None:
    """No training run has ever been recorded there, so there is nothing to score with.

    503, not 404: the campaign-day is not the thing that is missing, the
    models are, and a 404 would tell a caller to go look for a different day.
    """
    response = _score(empty_client, "cmp_0000_000", "2025-12-31")

    assert response.status_code == 503
    assert "artifact" in response.json()["detail"]


def test_the_models_are_loaded_once_and_shared(client, scoreable_day) -> None:
    """Loading per request would unpickle 200 trees plus two ensembles per call."""
    campaign_id, event_date = scoreable_day
    _score(client, campaign_id, event_date)
    first = client.app.state.scoring_models.get()
    _score(client, campaign_id, event_date)

    assert first is not None
    assert client.app.state.scoring_models.get() is first
