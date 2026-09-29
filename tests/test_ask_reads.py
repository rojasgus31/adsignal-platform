"""The database-level controls behind `/ask`, tested with the validator bypassed.

`tests/test_sql_guard.py` proves the validator refuses hostile SQL. This file
proves it would not matter if it did not: every statement below is handed
straight to the executor, or to a raw connection opened with the read-only
role, so nothing the validator does is in the path. If these tests pass, a
validator bug is a bug and not a breach.

They run against `ASK_TEST_DSN`, a database of their own, because a
regression here means a `DROP TABLE` actually lands.
"""

import os
import secrets
from datetime import date

import psycopg
import pytest

from adsignal.warehouse.artifact_store import ensure_artifact_schema
from adsignal.warehouse.ask_reads import (
    READONLY_ROLE,
    STATEMENT_TIMEOUT,
    QueryRefused,
    ensure_readonly_role,
    forget_verified_dsns,
    readonly_dsn_for,
    run_readonly_query,
    verify_readonly_dsn,
)
from adsignal.warehouse.loader import ensure_schema as ensure_mart_schema
from adsignal.warehouse.metrics_store import ensure_ml_schema

# The same database `conftest.py` creates under this name; every DSN-driven
# test module in this suite reads its own DSN from the environment this way.
ASK_TEST_DSN = os.environ.get(
    "ADSIGNAL_ASK_TEST_DSN",
    "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_ask",
)

pytestmark = pytest.mark.usefixtures("readonly_dsn")


def _postgres_available(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


@pytest.fixture(scope="module")
def readonly_password() -> str:
    """The password this module's `adsignal_ro` is given.

    Generated per run rather than written down anywhere: the role is a local
    test artefact, and a literal in the repository would be a credential in
    the repository. Shared by every test in the module, because the role is
    cluster-wide — a test that rotated it to a different value would break
    every direct `psycopg.connect` that ran after it.
    """
    return secrets.token_urlsafe(24)


@pytest.fixture(scope="module")
def readonly_dsn(readonly_password: str) -> str:
    """Create `adsignal_ro` against `ASK_TEST_DSN` and return its DSN."""
    if not _postgres_available(ASK_TEST_DSN):
        pytest.skip("Postgres not reachable")

    ensure_mart_schema(ASK_TEST_DSN)
    ensure_ml_schema(ASK_TEST_DSN)
    ensure_artifact_schema(ASK_TEST_DSN)

    ensure_readonly_role(ASK_TEST_DSN, readonly_password)
    return readonly_dsn_for(ASK_TEST_DSN, readonly_password)


def _table_exists(table: str) -> bool:
    with psycopg.connect(ASK_TEST_DSN) as conn:
        row = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()
    assert row is not None
    return row[0] is not None


def test_the_role_itself_refuses_a_drop(readonly_dsn: str) -> None:
    """Control 1, on its own: no validator, no read-only transaction, just the role.

    This connects with `adsignal_ro` directly and issues `DROP TABLE` in an
    ordinary read-write transaction. The role holds SELECT on seven tables
    and owns nothing, so Postgres refuses it. This is the test that says the
    guarantee lives in the database rather than in a string scanner.
    """
    assert _table_exists("dim_campaign")
    with (
        psycopg.connect(readonly_dsn) as conn,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        conn.execute("DROP TABLE dim_campaign")
    assert _table_exists("dim_campaign"), "the read-only role dropped a table"


def test_the_role_itself_refuses_an_insert(readonly_dsn: str) -> None:
    with (
        psycopg.connect(readonly_dsn) as conn,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        conn.execute("INSERT INTO dim_campaign VALUES ('x', 'x', 'x', 'x', 'x')")


def test_the_role_itself_cannot_create_a_table(readonly_dsn: str) -> None:
    with (
        psycopg.connect(readonly_dsn) as conn,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        conn.execute("CREATE TABLE adsignal_ro_should_not_have_made_this (x int)")


def test_the_executor_refuses_a_drop_the_validator_never_saw(readonly_dsn: str) -> None:
    """The brief's step 4: call the executor directly with a destructive statement.

    `run_readonly_query` does not validate; `routes/ask.py` validates before
    calling it. Passing `DROP TABLE` in here is therefore the exact "what if
    the validator is bypassed" case, and it must come back as a typed refusal
    with the table still present.
    """
    assert _table_exists("dim_campaign")
    with pytest.raises(QueryRefused) as refused:
        run_readonly_query(readonly_dsn, "DROP TABLE dim_campaign")
    assert refused.value.rule in ("database_readonly", "database_permission")
    assert _table_exists("dim_campaign"), "the executor dropped a table"


def test_the_executor_refuses_an_update(readonly_dsn: str) -> None:
    with pytest.raises(QueryRefused) as refused:
        run_readonly_query(readonly_dsn, "UPDATE dim_campaign SET channel = 'x'")
    assert refused.value.rule in ("database_readonly", "database_permission")


def test_the_excluded_tables_are_unreadable_by_the_role(readonly_dsn: str) -> None:
    """`model_artifacts` and `entity_embeddings` are revoked, not merely unmentioned."""
    for table in ("model_artifacts", "entity_embeddings"):
        if not _table_exists(table):
            continue
        with pytest.raises(QueryRefused) as refused:
            run_readonly_query(readonly_dsn, f"SELECT * FROM {table} LIMIT 1")  # noqa: S608
        assert refused.value.rule == "database_permission"


def test_an_allowlisted_select_returns_columns_and_rows(readonly_dsn: str) -> None:
    with psycopg.connect(ASK_TEST_DSN) as conn:
        conn.execute("DELETE FROM dim_campaign WHERE campaign_id = %s", ("cmp_ro_test",))
        conn.execute(
            "INSERT INTO dim_campaign VALUES (%s, %s, %s, %s, %s)",
            ("cmp_ro_test", "Readonly Test", "adv_ro", "Readonly Advertiser", "search"),
        )
        conn.commit()

    result = run_readonly_query(
        readonly_dsn,
        "SELECT campaign_id, campaign_name FROM dim_campaign WHERE campaign_id = 'cmp_ro_test'",
    )
    assert result.columns == ("campaign_id", "campaign_name")
    assert result.rows == [{"campaign_id": "cmp_ro_test", "campaign_name": "Readonly Test"}]


def test_rows_come_back_json_safe(readonly_dsn: str) -> None:
    """A DATE reaches the response as a string, not as a Python object."""
    with psycopg.connect(ASK_TEST_DSN) as conn:
        conn.execute(
            "INSERT INTO fct_campaign_daily (event_date, campaign_id, advertiser_id, channel, "
            "impressions, clicks, conversions, spend, revenue, is_anomaly) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (date(2024, 5, 1), "cmp_ro_test", "adv_ro", "search", 10, 1, 1, 2.5, 5.0, False),
        )
        conn.commit()

    result = run_readonly_query(
        readonly_dsn,
        "SELECT event_date, spend FROM fct_campaign_daily WHERE campaign_id = 'cmp_ro_test'",
    )
    assert result.rows[0]["event_date"] == "2024-05-01"
    assert isinstance(result.rows[0]["spend"], float)


def test_a_runaway_query_dies_on_the_statement_timeout(readonly_dsn: str) -> None:
    """Control 2: a query nothing else stops is killed by the database clock.

    The timeout is overridden to keep the suite fast; the default this
    endpoint actually runs with is asserted separately below.
    """
    with pytest.raises(QueryRefused) as refused:
        run_readonly_query(readonly_dsn, "SELECT pg_sleep(10)", statement_timeout="250ms")
    assert refused.value.rule == "statement_timeout"


def test_the_shipped_timeout_is_five_seconds() -> None:
    assert STATEMENT_TIMEOUT == "5s"


def test_a_missing_table_is_a_refusal_not_a_crash(readonly_dsn: str) -> None:
    with pytest.raises(QueryRefused) as refused:
        run_readonly_query(readonly_dsn, "SELECT * FROM no_such_table_at_all")
    assert refused.value.rule in ("database_permission", "undefined")


def test_a_refusal_carries_no_dsn_no_password_and_no_driver_text(readonly_dsn: str) -> None:
    """Control 5: a refused query explains the rule and leaks nothing else."""
    with pytest.raises(QueryRefused) as refused:
        run_readonly_query(readonly_dsn, "DROP TABLE dim_campaign")
    text = f"{refused.value.reason} {refused.value.rule}"
    password = psycopg.conninfo.conninfo_to_dict(readonly_dsn)["password"]
    assert isinstance(password, str)
    assert password not in text
    assert "postgresql://" not in text
    assert "localhost" not in text
    assert "psycopg" not in text
    assert "Traceback" not in text
    assert str(refused.value) == refused.value.reason


def test_ensure_readonly_role_is_idempotent(readonly_password: str) -> None:
    """Running it twice, as every deploy does, must succeed and leave it usable.

    Deliberately the module's own password, not a fresh one: `adsignal_ro` is
    cluster-wide, so rotating it here would invalidate the DSN every other
    test in this module holds — which is exactly what happened, and was only
    visible once a later test opened a direct connection instead of reusing
    a pooled one.
    """
    if not _postgres_available(ASK_TEST_DSN):
        pytest.skip("Postgres not reachable")
    ensure_readonly_role(ASK_TEST_DSN, readonly_password)
    ensure_readonly_role(ASK_TEST_DSN, readonly_password)
    result = run_readonly_query(
        readonly_dsn_for(ASK_TEST_DSN, readonly_password),
        "SELECT 1 AS one FROM dim_campaign LIMIT 1",
    )
    assert result.columns == ("one",)


def test_the_role_has_select_on_exactly_the_allowlisted_tables(readonly_dsn: str) -> None:
    """The grant and the allowlist are the same set, checked against the catalog.

    Equality, not containment: `model_artifacts` and `entity_embeddings`
    being absent is the point, and a subset assertion plus two named
    absences would miss any third table someone grants later.
    """
    from adsignal.warehouse.ask_schema import ALLOWED_TABLES

    with psycopg.connect(ASK_TEST_DSN) as conn:
        granted = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.role_table_grants "
                "WHERE grantee = %s AND privilege_type = 'SELECT'",
                (READONLY_ROLE,),
            ).fetchall()
        }
    # Exactly the allowlist, not a superset: a grant the allowlist does not
    # name is a table the validator would refuse but the role could still
    # read if the validator were bypassed.
    assert granted == set(ALLOWED_TABLES)


# Fix round 1, finding 3. Control 1 is only a control if something checks
# that the DSN generated SQL runs on is the restricted role. Pasting the
# admin DSN into `ADSIGNAL_POSTGRES_READONLY_DSN` silently deleted it.


def test_the_executor_refuses_an_unrestricted_dsn(readonly_dsn: str) -> None:
    """The admin DSN in the read-only slot must be refused, not used.

    `ASK_TEST_DSN` is the admin connection. Handing it to the executor is
    exactly the misconfiguration this check exists for, and the query below
    is a harmless SELECT the admin role could perfectly well run — the point
    is that it must not be allowed to.
    """
    with pytest.raises(QueryRefused) as refused:
        run_readonly_query(ASK_TEST_DSN, "SELECT campaign_id FROM dim_campaign LIMIT 1")
    assert refused.value.rule == "readonly_dsn_unverified"
    assert "postgresql://" not in refused.value.reason
    assert "adsignal" not in refused.value.reason.lower()


def test_the_restricted_dsn_passes_verification(readonly_dsn: str) -> None:
    verify_readonly_dsn(readonly_dsn)


def test_verification_rejects_a_role_that_can_write(readonly_dsn: str) -> None:
    with pytest.raises(QueryRefused) as refused:
        verify_readonly_dsn(ASK_TEST_DSN)
    assert refused.value.rule == "readonly_dsn_unverified"


def test_the_fetch_is_bounded_even_if_the_query_is_not(readonly_dsn: str) -> None:
    """Fix round 1, finding 2: `fetchall` on an uncapped query is unbounded memory.

    The validator caps the outer query, so this only happens if the validator
    is bypassed — which is precisely the case the executor must survive.
    """
    result = run_readonly_query(readonly_dsn, "SELECT campaign_id FROM dim_campaign", max_rows=3)
    assert len(result.rows) <= 3


# Fix round 2. Two controls below the rules: the wire protocol, and not
# retaining a credential in a process-global structure.


def test_a_stacked_statement_is_refused_by_the_wire_protocol(readonly_dsn: str) -> None:
    """Control 2, at a level no rule of ours is involved in.

    The validator refuses stacked statements, but it is bypassable by
    definition — that is why the executor is tested separately. psycopg sends
    a parameterless query over the *simple* protocol, where Postgres happily
    runs several statements separated by semicolons; the executor now forces
    the extended protocol, where the server refuses them itself.

    This asserts the refusal comes from Postgres, not from us: the statement
    below never touches `validate_sql`.
    """
    with pytest.raises(QueryRefused) as refused:
        run_readonly_query(readonly_dsn, "SELECT 1 AS a; SELECT 2 AS b")
    assert refused.value.rule in ("invalid_query", "database_error")


def test_the_simple_protocol_would_have_allowed_that_stack(readonly_dsn: str) -> None:
    """The counter-test, so the one above cannot pass for the wrong reason.

    Run the identical statement on a plain connection with no `prepare`, and
    Postgres accepts it. If this ever starts failing, the test above stopped
    proving anything about the protocol and started proving something about
    the SQL.
    """
    with psycopg.connect(readonly_dsn) as conn:
        rows = conn.execute("SELECT 1 AS a; SELECT 2 AS b").fetchall()
    assert rows == [(1,)], "the simple protocol no longer accepts a stacked statement"


def test_the_verification_memo_does_not_retain_the_dsn(readonly_dsn: str) -> None:
    """A DSN carries the role's password; a process-global set must not hold it."""
    from adsignal.warehouse import ask_reads

    forget_verified_dsns()
    verify_readonly_dsn(readonly_dsn)
    remembered = " ".join(ask_reads._VERIFIED_DSNS)  # noqa: SLF001 - the point of the test
    password = psycopg.conninfo.conninfo_to_dict(readonly_dsn)["password"]
    assert isinstance(password, str)
    assert password not in remembered
    assert readonly_dsn not in remembered
    assert "postgresql://" not in remembered
    assert remembered, "nothing was memoized, so the check was not exercised"
