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
3. `QueryRefused` is how any database failure reaches the caller: a short
   reason and a rule name, never a DSN, a password or driver text.

Nothing here validates SQL — `warehouse.sql_guard` does that, and
`api.routes.ask` calls it first. Keeping them apart is what lets
`tests/test_ask_reads.py` hand this module a `DROP TABLE` directly and prove
the database refuses it on its own.
"""

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

# The login role generated SQL runs as. One name, used by the DDL below and
# by whoever writes `ADSIGNAL_POSTGRES_READONLY_DSN`.
READONLY_ROLE = "adsignal_ro"

# The per-statement deadline. A cross join the validator happily accepts can
# still run for hours; this is what kills it. Short enough that a stuck query
# cannot hold a pooled connection while a dashboard waits on it.
STATEMENT_TIMEOUT = "5s"

# Revoked rather than merely left ungranted. A future `GRANT SELECT ON ALL
# TABLES IN SCHEMA public` run by someone else would otherwise hand the role
# pickled estimator bytes and raw embedding vectors.
EXCLUDED_TABLES = ("model_artifacts", "entity_embeddings")


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


def run_readonly_query(
    readonly_dsn: str, query: str, statement_timeout: str = STATEMENT_TIMEOUT
) -> QueryResult:
    """Run one already-validated statement as the read-only role.

    `readonly_dsn` must be the `adsignal_ro` DSN; the admin DSN is never
    passed here, and `api.routes.ask` reads it from its own setting rather
    than deriving it. Inside the transaction, `SET LOCAL
    transaction_read_only = on` blocks writes even for a role that somehow
    holds them, and `SET LOCAL statement_timeout` bounds the runtime. Both
    are `LOCAL`, so they end with the transaction and cannot leak onto the
    next caller of a pooled connection.

    This function does **not** validate `query`. Every caller in the
    application validates first; the tests deliberately do not, because a
    control that only works when the validator agrees is not a control.
    """
    try:
        with connection(readonly_dsn) as conn, conn.transaction():
            conn.execute("SET LOCAL transaction_read_only = on")
            conn.execute(
                sql.SQL("SET LOCAL statement_timeout = {}").format(sql.Literal(statement_timeout))
            )
            cursor = conn.execute(query)
            columns = tuple(column.name for column in cursor.description or ())
            fetched = cursor.fetchall()
    except psycopg.Error as error:
        raise _refusal(error) from None
    return QueryResult(columns=columns, rows=[_row_to_dict(columns, row) for row in fetched])


def _refusal(error: psycopg.Error) -> QueryRefused:
    """Translate a driver error into a refusal that carries none of its text."""
    if isinstance(error, psycopg.errors.InsufficientPrivilege):
        return QueryRefused(
            "the read-only role is not allowed to touch that", "database_permission"
        )
    if isinstance(error, psycopg.errors.ReadOnlySqlTransaction):
        return QueryRefused("this connection may only read", "database_readonly")
    if isinstance(error, psycopg.errors.QueryCanceled):
        return QueryRefused(
            f"the query ran longer than {STATEMENT_TIMEOUT} and was cancelled",
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
