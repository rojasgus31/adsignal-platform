"""Serialising the `ensure_*_schema` helpers against each other.

Every read path runs its `ensure_*_schema` helper before it queries, so a
database the pipeline has never run against still answers with an empty
result instead of a database error. Those helpers issue `CREATE TABLE IF NOT
EXISTS` / `CREATE EXTENSION IF NOT EXISTS` as ordinary DDL, and Postgres does
**not** serialise concurrent `IF NOT EXISTS` DDL: the losers of the race abort
with `UniqueViolation` on a `pg_catalog` index (`pg_type_typname_nsp_index`,
`pg_class_relname_nsp_index`) or with `DuplicateTable`. On a fresh
deployment's first page load, where the dashboard fans out to four endpoints
at once, most of those requests fail.

`apply_schema` is the one mechanism all four helpers share. A
transaction-scoped Postgres advisory lock makes the DDL mutually exclusive
across connections, processes and containers — the whole database cluster,
not just this process — and is released by the commit (or by the rollback an
error triggers), so no path can leak it. Catching `UniqueViolation` and
calling it success would also stop the 500s, but it would leave the losing
connection's transaction aborted mid-DDL, with the remaining statements
unapplied, which is a weaker guarantee than never racing at all.
"""

import psycopg

# An arbitrary but fixed 64-bit key naming "adsignal schema DDL" among all
# advisory locks taken on this database. It must never change: two versions of
# this code using two keys would not exclude each other.
SCHEMA_LOCK_KEY = 0x4144_5347_4E4C_0001


def apply_schema(dsn: str, *statements: str) -> None:
    """Run DDL `statements` in order, under the shared schema advisory lock.

    The lock is transaction-scoped, so it is held from here until the commit
    below and no caller has to release it.
    """
    with psycopg.connect(dsn) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK_KEY,))
        for statement in statements:
            conn.execute(statement)
        conn.commit()
