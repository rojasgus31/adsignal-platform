"""What the model is told about the corpus, and what an empty answer says.

The defect these cover: `SCHEMA_DESCRIPTION` named tables and columns and
said nothing about *when*. Asked the dashboard's own suggested question -
"which channel spent the most last month?" - Gemini wrote a correct
`CURRENT_DATE`-relative filter over a warehouse whose last row predates it,
and the endpoint answered zero rows.

Everything here is pure: no database, no network, no environment.
"""

from datetime import date

import pytest

from adsignal.warehouse.ask_schema import (
    COVERAGE_TABLE,
    DATE_COLUMN,
    SCHEMA_DESCRIPTION,
    DateCoverage,
    describe_schema,
    empty_result_notice,
)

COVERAGE = DateCoverage(first_date=date(2024, 1, 1), last_date=date(2025, 12, 31))


def test_the_description_states_the_range_the_data_actually_covers() -> None:
    described = describe_schema(COVERAGE)
    assert "2024-01-01" in described
    assert "2025-12-31" in described


def test_the_description_says_the_corpus_does_not_reach_today() -> None:
    """The fact the model could not infer, and the one that caused the defect."""
    described = describe_schema(COVERAGE).lower()
    assert "historical" in described
    assert "does not\nreach today's date" in described


def test_relative_expressions_are_anchored_to_the_latest_date_not_to_now() -> None:
    described = describe_schema(COVERAGE)
    assert "last month" in described
    assert "CURRENT_DATE" in described
    assert "NOW()" in described
    assert "DATE '2025-12-31'" in described


def test_the_table_shapes_survive_the_coverage_paragraph() -> None:
    """Grounding is appended to the schema, not substituted for it."""
    assert SCHEMA_DESCRIPTION in describe_schema(COVERAGE)
    assert SCHEMA_DESCRIPTION in describe_schema(None)


def test_an_empty_warehouse_still_produces_a_usable_description() -> None:
    """No rows is a state, not a crash: the prompt says so in words."""
    described = describe_schema(None)
    assert "no rows" in described
    assert COVERAGE_TABLE in described
    assert DATE_COLUMN in described


def test_an_empty_warehouse_description_invents_no_dates() -> None:
    """Falling back to `config.start_date` here is the drift this avoids."""
    described = describe_schema(None)
    assert "2024-01-01" not in described
    assert "2025-12-31" not in described


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT channel FROM fct_campaign_daily WHERE event_date >= DATE '2026-09-01' LIMIT 5",
        "SELECT EVENT_DATE FROM fct_campaign_daily LIMIT 5",
        "SELECT SUM(spend) FROM fct_campaign_daily GROUP BY event_date LIMIT 5",
    ],
)
def test_a_query_that_names_the_date_column_gets_the_coverage_sentence(sql: str) -> None:
    notice = empty_result_notice(sql, COVERAGE)
    assert notice is not None
    assert "2024-01-01" in notice
    assert "2025-12-31" in notice


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT campaign_id FROM dim_campaign WHERE channel = 'search' LIMIT 5",
        "SELECT run_id FROM model_metrics LIMIT 5",
        "SELECT event_dates FROM dim_campaign LIMIT 5",
    ],
)
def test_a_query_that_does_not_filter_on_a_date_gets_no_notice(sql: str) -> None:
    """Silence beats inventing a date explanation for a question about names."""
    assert empty_result_notice(sql, COVERAGE) is None


def test_an_empty_warehouse_explains_an_empty_answer_too() -> None:
    notice = empty_result_notice("SELECT 1 FROM fct_campaign_daily WHERE event_date > x", None)
    assert notice is not None
    assert "no rows" in notice
