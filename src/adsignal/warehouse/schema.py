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

import threading

import psycopg

# An arbitrary but fixed 64-bit key naming "adsignal schema DDL" among all
# advisory locks taken on this database. It must never change: two versions of
# this code using two keys would not exclude each other.
SCHEMA_LOCK_KEY = 0x4144_5347_4E4C_0001

# Which (dsn, statements) pairs this process has already applied. Every read
# path calls its `ensure_*_schema` helper before querying (see the docstrings
# on those helpers), so without this memo, every single warehouse call still
# paid for one fresh, unpooled connect here - on top of its pooled one for the
# actual query - regardless of `warehouse.pool`. The memo is what actually
# delivers the reduced-connections goal that module was written for.
_ENSURED: set[tuple[str, tuple[str, ...]]] = set()
# One lock per distinct (dsn, statements) key, not one global lock: a
# concurrent burst of first-ever requests for the *same* key must serialise
# on the connect (so exactly one of them performs it and the rest simply wait
# and then see the memo hit - see `apply_schema`), but a burst for a
# *different* key must not queue up behind that connect.
_ENSURED_LOCKS: dict[tuple[str, tuple[str, ...]], threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _lock_for(key: tuple[str, tuple[str, ...]]) -> threading.Lock:
    with _REGISTRY_LOCK:
        lock = _ENSURED_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _ENSURED_LOCKS[key] = lock
        return lock


def apply_schema(dsn: str, *statements: str) -> None:
    """Run DDL `statements` in order, under the shared schema advisory lock.

    The lock is transaction-scoped, so it is held from here until the commit
    below and no caller has to release it.

    Deliberately not routed through `warehouse.pool`: pooling this connection
    would risk the advisory lock still being held when another caller checks
    the same physical connection back out, since the lock's scope is this one
    transaction on this one connection, not the logical "this DSN" the pool
    keys on. A dedicated `psycopg.connect` per call keeps the lock's lifetime
    and the connection's lifetime identical.

    Memoized per (`dsn`, `statements`): once this exact DDL has been applied
    against a DSN in this process, later calls return immediately without
    connecting at all. A per-key lock (`_lock_for`), not just a guarded set,
    is what makes this hold under a concurrent burst of *first-ever* calls
    for the same key too - a fast-path check against the bare set would let
    every thread racing to be first see a miss and each open its own connect
    before any of them finished, which is exactly the "one dashboard page
    load, several connects" problem this module exists to close. Holding the
    lock across the connect below only serialises callers sharing this exact
    key; a different (dsn, statements) pair uses a different lock, so it is
    never stalled by this one - including during a Neon cold start, where
    this connect can take seconds.
    """
    key = (dsn, statements)
    with _REGISTRY_LOCK:
        if key in _ENSURED:
            return

    with _lock_for(key):
        with _REGISTRY_LOCK:
            if key in _ENSURED:
                return
        with psycopg.connect(dsn) as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK_KEY,))
            for statement in statements:
                conn.execute(statement)
            conn.commit()
        with _REGISTRY_LOCK:
            _ENSURED.add(key)


def reset_schema_memo(dsn: str | None = None) -> None:
    """Forget what this process has ensured for `dsn`, or for every DSN. Test-only.

    Lets a test that drops tables out from under a DSN (simulating a fresh
    deployment) force the next `ensure_*_schema` call for that DSN to really
    reapply the DDL, instead of trusting a memo an earlier test in the same
    process left behind. Scoped to one DSN by default so it does not also
    discard - and force a redundant DDL connect for - unrelated DSNs other
    fixtures in the same test session are using; pass no argument to clear
    every DSN at once.
    """
    with _REGISTRY_LOCK:
        if dsn is None:
            _ENSURED.clear()
            return
        _ENSURED.difference_update({key for key in _ENSURED if key[0] == dsn})
