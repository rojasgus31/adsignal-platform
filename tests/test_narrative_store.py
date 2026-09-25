import os
from datetime import date

import psycopg
import pytest

from adsignal.narrative.types import NarrativeResult
from adsignal.warehouse.narrative_store import (
    ensure_narrative_schema,
    read_latest_narrative,
    write_narrative,
)

DSN = os.environ.get(
    "ADSIGNAL_TEST_DSN", "postgresql://adsignal:adsignal@localhost:5432/adsignal_test"
)


@pytest.fixture(autouse=True)
def clean():
    ensure_narrative_schema(DSN)
    with psycopg.connect(DSN) as conn:
        conn.execute("TRUNCATE TABLE narrative_runs")
        conn.commit()
    yield


def test_round_trip() -> None:
    write_narrative(DSN, date(2024, 6, 1), NarrativeResult("all quiet", "gemini", 412.0))
    stored = read_latest_narrative(DSN)
    assert stored is not None
    assert stored.text == "all quiet"
    assert stored.provider == "gemini"
    assert stored.as_of == date(2024, 6, 1)


def test_latest_wins_when_several_exist() -> None:
    write_narrative(DSN, date(2024, 6, 1), NarrativeResult("older", "groq", 1.0))
    write_narrative(DSN, date(2024, 6, 2), NarrativeResult("newer", "gemini", 1.0))
    stored = read_latest_narrative(DSN)
    assert stored is not None and stored.text == "newer"


def test_history_is_kept_rather_than_overwritten() -> None:
    write_narrative(DSN, date(2024, 6, 1), NarrativeResult("a", "groq", 1.0))
    write_narrative(DSN, date(2024, 6, 2), NarrativeResult("b", "gemini", 1.0))
    with psycopg.connect(DSN) as conn:
        count = conn.execute("SELECT count(*) FROM narrative_runs").fetchone()[0]
    assert count == 2


def test_reading_an_empty_table_returns_none() -> None:
    assert read_latest_narrative(DSN) is None


def test_schema_creation_is_idempotent() -> None:
    ensure_narrative_schema(DSN)
    ensure_narrative_schema(DSN)


def test_stored_row_records_which_provider_answered() -> None:
    write_narrative(DSN, date(2024, 6, 1), NarrativeResult("text", "jev", 70.0))
    stored = read_latest_narrative(DSN)
    assert stored is not None and stored.provider == "jev"
    assert stored.latency_ms == pytest.approx(70.0)
