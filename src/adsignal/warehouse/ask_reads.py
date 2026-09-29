"""Running `/ask`'s generated SQL, and the role that makes doing so safe.

Three things live here, in the order they matter:

1. `ensure_readonly_role` creates `adsignal_ro`: a login role holding SELECT
   on the seven allowlisted tables and nothing else, with `model_artifacts`
   and `entity_embeddings` explicitly revoked. It owns no table, so DDL
   against any of them is refused by Postgres itself.
2. `run_readonly_query` executes one statement on a connection opened with
   that role, inside a transaction that sets `transaction_read_only` and a
   statement timeout. A query that gets past the validator still has to get
   past both of these.
3. `verify_readonly_dsn` proves, before the first query runs on a DSN, that
   the connection really is restricted — not a superuser, unable to write to
   an allowlisted table, unable to read an excluded one. Without it, an admin
   DSN pasted into the read-only setting would delete control 1 in silence.
4. `QueryRefused` is how any database failure reaches the caller: a short
   reason and a rule name, never a DSN, a password or driver text.

Nothing here validates SQL — `warehouse.sql_guard` does that, and
`api.routes.ask` calls it first. Keeping them apart is what lets
`tests/test_ask_reads.py` hand this module a `DROP TABLE` directly and prove
the database refuses it on its own.
"""

import hashlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg import sql

from adsignal.warehouse.artifact_store import ensure_artifact_schema
from adsignal.warehouse.ask_schema import ALLOWED_TABLES
from adsignal.warehouse.loader import ensure_schema
from adsignal.warehouse.metrics_store import ensure_ml_schema
from adsignal.warehouse.narrative_store import ensure_narrative_schema
from adsignal.warehouse.pool import connection
from adsignal.warehouse.run_log import ensure_run_log_schema
from adsignal.warehouse.schema import SCHEMA_LOCK_KEY
from adsignal.warehouse.sql_guard import MAX_ROWS

# The login role generated SQL runs as. One name, used by the DDL below and
# by whoever writes `ADSIGNAL_POSTGRES_READONLY_DSN`.
READONLY_ROLE = "adsignal_ro"

# The per-statement deadline. A cross join the validator happily accepts can
# still run for hours; this is what kills it. Short enough that a stuck query
# cannot hold a pooled connection while a dashboard waits on it.
STATEMENT_TIMEOUT = "5s"

# Server settings the validator's correctness silently depends on, pinned for
# the duration of every query so that the SQL Postgres reads means what
# `sql_guard` decided it means.
#
# `standard_conforming_strings` is the one that matters. With it `off`,
# Postgres reads a backslash in an ordinary string literal as an escape, so
# `SELECT '\' AS a, ', 1 AS z FROM model_artifacts --' AS c FROM
# dim_campaign` - approved as reading `dim_campaign` - reads
# `model_artifacts` instead. sqlglot's Postgres dialect always assumes `on`,
# so the round-trip check structurally *cannot* see this divergence: both
# parses agree with each other and disagree with the server. The default is
# `on`, but `postgresql.conf` or `ALTER ROLE adsignal_ro SET ...` can change
# it invisibly, and a control whose premise is unverified is the same defect
# as a control nobody verifies.
#
# `backslash_quote = off` refuses `\'` as an escaped quote inside `E''`
# strings outright. That is the exact construction behind the escape-string
# regression, and nothing a question about campaign performance needs.
#
# `search_path = public` makes the table allowlist mean what it says: the
# validator refuses schema-qualified names, so every name it approves is
# resolved by the search path, and leaving that to the role's configuration
# would let `dim_campaign` mean a table in some other schema.
_PINNED_SETTINGS = (
    ("standard_conforming_strings", "on"),
    ("backslash_quote", "off"),
    ("search_path", "public"),
)

# Revoked rather than merely left ungranted. A future `GRANT SELECT ON ALL
# TABLES IN SCHEMA public` run by someone else would otherwise hand the role
# pickled estimator bytes and raw embedding vectors.
EXCLUDED_TABLES = ("model_artifacts", "entity_embeddings")


# Asked of the read-only connection before the first query runs on it. A
# privilege check rather than an attempted write: `has_table_privilege`
# accounts for superuser status, role membership and default grants, and it
# touches no row, so verification cannot itself change anything.
_PRIVILEGE_PROBE = """
SELECT
    (SELECT rolsuper FROM pg_roles WHERE rolname = current_user),
    CASE WHEN to_regclass(%(allowed)s) IS NULL THEN NULL
         ELSE has_table_privilege(%(allowed)s, 'INSERT, UPDATE, DELETE') END,
    CASE WHEN to_regclass(%(excluded)s) IS NULL THEN NULL
         ELSE has_table_privilege(%(excluded)s, 'SELECT') END
"""

# The table the probe tests for write access (always created by
# `ensure_schema`) and the one it tests for read access (never granted).
_PROBE_ALLOWED_TABLE = "dim_campaign"
_PROBE_EXCLUDED_TABLE = "model_artifacts"

# DSNs already proven restricted, so the probe costs one round trip per DSN
# per process rather than one per question. Only successes are remembered: a
# DSN that failed verification must be re-checked, because the fix is a grant
# change that this process would otherwise never notice.
#
# Keyed by digest, not by the DSN itself. A DSN carries the role's password,
# and this set lives for the life of the process — retaining it here would be
# exactly what `ensure_readonly_role` avoids by not using `apply_schema`'s
# memo. A digest answers "have I checked this one" without holding the
# credential to do it.
_VERIFIED_DSNS: set[str] = set()
_VERIFY_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class QueryResult:
    """One generated query's result: its column names and its JSON-safe rows."""

    columns: tuple[str, ...]
    rows: list[dict[str, Any]]


class QueryRefused(RuntimeError):
    """A query the database would not run, in terms safe to show a user.

    Every `psycopg` error is converted to one of these before it leaves this
    module. The driver's own message can carry the failing statement, the
    connection's parameters and a chained traceback, none of which belongs in
    an API response, so none of it is carried across.
    """

    def __init__(self, reason: str, rule: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.rule = rule


def ensure_readonly_role(admin_dsn: str, ro_password: str, role: str = READONLY_ROLE) -> None:
    """Create or update `role` as a SELECT-only login on the allowlisted tables.

    Idempotent: a missing role is created, an existing one has its password
    reset and its grants reapplied, so running this on every deploy is the
    intended usage. The allowlisted tables are ensured first, because a GRANT
    naming a table that does not exist yet is an error and a fresh database
    has none of them.

    Held under the same transaction-scoped advisory lock as every other DDL
    path in this package (`schema.SCHEMA_LOCK_KEY`), so two deploys racing
    each other serialise instead of colliding on `CREATE ROLE`. It does not
    go through `apply_schema` itself: that helper memoizes the statements it
    ran, and these statements contain a password, which must not be retained
    in a process-global set.

    `ro_password` is written into the role with `psycopg.sql.Literal` and is
    never logged, echoed, returned or included in any error raised here.
    """
    ensure_schema(admin_dsn)
    ensure_ml_schema(admin_dsn)
    ensure_narrative_schema(admin_dsn)
    ensure_run_log_schema(admin_dsn)
    ensure_artifact_schema(admin_dsn)

    role_name = sql.Identifier(role)
    database = _database_name(admin_dsn)
    with psycopg.connect(admin_dsn) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK_KEY,))
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
        verb = sql.SQL("ALTER ROLE") if exists is not None else sql.SQL("CREATE ROLE")
        conn.execute(
            sql.SQL("{verb} {role} WITH LOGIN PASSWORD {password}").format(
                verb=verb, role=role_name, password=sql.Literal(ro_password)
            )
        )
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {database} TO {role}").format(
                database=sql.Identifier(database), role=role_name
            )
        )
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {role}").format(role=role_name))
        for table in sorted(ALLOWED_TABLES):
            conn.execute(
                sql.SQL("GRANT SELECT ON {table} TO {role}").format(
                    table=sql.Identifier(table), role=role_name
                )
            )
        for table in EXCLUDED_TABLES:
            if not _table_exists(conn, table):
                # `entity_embeddings` only exists where pgvector is
                # installed. Nothing was granted on it, so there is nothing
                # to revoke; the next run will revoke it once it exists.
                continue
            conn.execute(
                sql.SQL("REVOKE ALL ON {table} FROM {role}").format(
                    table=sql.Identifier(table), role=role_name
                )
            )
        conn.commit()


def readonly_dsn_for(admin_dsn: str, ro_password: str, role: str = READONLY_ROLE) -> str:
    """Return `admin_dsn`'s host and database with the read-only role's credentials.

    A convenience for deployment scripts and tests that have just called
    `ensure_readonly_role`: it keeps the two DSNs pointing at the same
    database by construction. Production reads its read-only DSN from
    `Settings.postgres_readonly_dsn` instead, and never builds one here.
    """
    # `conninfo_to_dict` types its values as `str | int | None`, while
    # `make_conninfo` takes strings; every value a DSN can carry renders as
    # one, so they are stringified here rather than suppressed.
    info = {
        key: str(value)
        for key, value in psycopg.conninfo.conninfo_to_dict(admin_dsn).items()
        if value is not None
    }
    info["user"] = role
    info["password"] = ro_password
    return psycopg.conninfo.make_conninfo(**info)


def verify_readonly_dsn(readonly_dsn: str) -> None:
    """Prove `readonly_dsn` really is the restricted role, or raise `QueryRefused`.

    Control 1 is a read-only database role. Nothing previously checked that
    the DSN generated SQL runs on *is* that role: pasting the admin DSN into
    `ADSIGNAL_POSTGRES_READONLY_DSN` deleted the control silently, and left
    every other control to hold on its own. A control nobody verifies is a
    control that might not be there.

    Three ways to fail, any one of which means this is not the read-only
    role: the connection is a superuser, it can write to an allowlisted
    table, or it can read one of the excluded ones. The check reads
    `has_table_privilege` and writes nothing.

    Memoized per DSN after it passes, so a healthy deployment pays one round
    trip per process.
    """
    fingerprint = _dsn_fingerprint(readonly_dsn)
    if fingerprint in _VERIFIED_DSNS:
        return

    try:
        with connection(readonly_dsn) as conn:
            row = conn.execute(
                _PRIVILEGE_PROBE,
                {"allowed": _PROBE_ALLOWED_TABLE, "excluded": _PROBE_EXCLUDED_TABLE},
            ).fetchone()
    except psycopg.Error:
        raise QueryRefused(
            "the read-only database connection could not be checked",
            "readonly_dsn_unverified",
        ) from None

    if row is None:  # pragma: no cover - a one-row probe always returns a row
        raise QueryRefused(
            "the read-only database connection could not be checked",
            "readonly_dsn_unverified",
        )

    is_superuser, can_write, can_read_excluded = row
    if is_superuser or can_write or can_read_excluded:
        # Deliberately says nothing about which check failed, which role was
        # used or which host it reached: this text is rendered to whoever
        # asked a question. The operator's copy of this is the grant itself.
        raise QueryRefused(
            "ask is misconfigured: its database connection is not the read-only role",
            "readonly_dsn_unverified",
        )

    with _VERIFY_LOCK:
        _VERIFIED_DSNS.add(fingerprint)


def _dsn_fingerprint(dsn: str) -> str:
    """A digest identifying `dsn` without retaining it.

    SHA-256 of the DSN. Not a secret-hiding measure against an attacker who
    already has the digest and a guessable DSN — it is there so the process
    never holds the password in a long-lived structure at all.
    """
    return hashlib.sha256(dsn.encode("utf-8")).hexdigest()


def forget_verified_dsns() -> None:
    """Forget which DSNs passed verification. Test-only.

    A test that changes a role's grants mid-session needs the next call to
    re-probe rather than trust a memo taken before the change.
    """
    with _VERIFY_LOCK:
        _VERIFIED_DSNS.clear()


def run_readonly_query(
    readonly_dsn: str,
    query: str,
    statement_timeout: str = STATEMENT_TIMEOUT,
    max_rows: int = MAX_ROWS,
) -> QueryResult:
    """Run one already-validated statement as the read-only role.

    `readonly_dsn` must be the `adsignal_ro` DSN; the admin DSN is never
    passed here, and `api.routes.ask` reads it from its own setting rather
    than deriving it. Inside the transaction, `SET LOCAL
    transaction_read_only = on` blocks writes even for a role that somehow
    holds them, `SET LOCAL statement_timeout` bounds the runtime, and
    `_PINNED_SETTINGS` fixes the parsing settings the validator assumed when
    it approved the query. All are `LOCAL`, so they end with the transaction
    and cannot leak onto the next caller of a pooled connection.

    This function does **not** validate `query`. Every caller in the
    application validates first; the tests deliberately do not, because a
    control that only works when the validator agrees is not a control.

    Executed on the extended query protocol (see the `prepare=True` comment
    below), so Postgres refuses a stacked statement before this module's
    rules are even consulted.

    `max_rows` bounds the fetch itself. The validator already caps the outer
    query, so this only bites when the validator was bypassed — which is the
    one case this function is written to survive, and `fetchall()` on an
    uncapped query is the API process's memory.
    """
    verify_readonly_dsn(readonly_dsn)
    try:
        with connection(readonly_dsn) as conn, conn.transaction():
            conn.execute("SET LOCAL transaction_read_only = on")
            conn.execute(
                sql.SQL("SET LOCAL statement_timeout = {}").format(sql.Literal(statement_timeout))
            )
            for name, value in _PINNED_SETTINGS:
                conn.execute(
                    sql.SQL("SET LOCAL {name} = {value}").format(
                        name=sql.Identifier(name), value=sql.Literal(value)
                    )
                )
            # `prepare=True` is what forces psycopg onto the **extended**
            # query protocol. Without it, a parameterless statement goes out
            # over the simple protocol, where Postgres itself accepts
            # several statements separated by semicolons — so
            # `_single_statement` would be the only thing standing between a
            # stacked payload and execution. On the extended protocol the
            # server refuses multiple commands outright, which makes stacking
            # impossible at the wire level rather than by rule. It also means
            # Postgres parses the statement with its own grammar, which is
            # the grammar that ultimately matters.
            cursor = conn.execute(query, prepare=True)
            columns = tuple(column.name for column in cursor.description or ())
            fetched = cursor.fetchmany(max_rows)
    except psycopg.Error as error:
        raise _refusal(error, statement_timeout) from None
    return QueryResult(columns=columns, rows=[_row_to_dict(columns, row) for row in fetched])


def _refusal(error: psycopg.Error, statement_timeout: str) -> QueryRefused:
    """Translate a driver error into a refusal that carries none of its text.

    `statement_timeout` is the value this call actually ran with, not the
    module default: a caller that shortened it must not be told the query
    survived five seconds when it was cancelled after 250 ms.
    """
    if isinstance(error, psycopg.errors.InsufficientPrivilege):
        return QueryRefused(
            "the read-only role is not allowed to touch that", "database_permission"
        )
    if isinstance(error, psycopg.errors.ReadOnlySqlTransaction):
        return QueryRefused("this connection may only read", "database_readonly")
    if isinstance(error, psycopg.errors.QueryCanceled):
        return QueryRefused(
            f"the query ran longer than {statement_timeout} and was cancelled",
            "statement_timeout",
        )
    if isinstance(error, psycopg.errors.UndefinedTable | psycopg.errors.UndefinedColumn):
        return QueryRefused("the query named a table or column that does not exist", "undefined")
    if isinstance(error, psycopg.errors.SyntaxError):
        return QueryRefused("the database could not parse that query", "invalid_query")
    return QueryRefused("the database refused that query", "database_error")


def _row_to_dict(columns: Sequence[str], row: Sequence[object]) -> dict[str, Any]:
    """Pair one result row with its column names, as JSON-safe values."""
    return {name: _jsonable(value) for name, value in zip(columns, row, strict=True)}


def _jsonable(value: object) -> Any:
    """Return `value` as something a JSON response can carry.

    A generated query can select any column of any allowlisted table, so this
    sees dates, timestamps and numerics that FastAPI would otherwise have to
    guess at inside a `dict[str, Any]`. Converting here makes the response's
    value types a fixed, documented set: string, number, boolean or null.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _database_name(dsn: str) -> str:
    """The database `dsn` connects to, for the GRANT CONNECT above."""
    name = psycopg.conninfo.conninfo_to_dict(dsn).get("dbname")
    if not isinstance(name, str) or not name:
        raise ValueError("the admin DSN does not name a database")
    return name


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    """Whether `table` exists in this database, without raising if it does not."""
    row = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()
    return row is not None and row[0] is not None
