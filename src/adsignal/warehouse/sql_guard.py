"""The pure validator standing between a generated query and the database.

`/ask` runs SQL a language model wrote. A model instructed to emit only
SELECT is not a security control, so this module assumes the model is
hostile and checks what it produced. It is one of three controls; the other
two — a read-only Postgres role and a read-only transaction with a statement
timeout, both in `warehouse.ask_reads` — hold even if this module is
bypassed entirely.

**This was a regex scanner and is now a parser, because the scanner was
wrong.** Three inputs reached `model_artifacts`, one of the two tables the
allowlist exists to protect: `FROM "model_artifacts"` (a quoted identifier
never matched the FROM pattern), `FROM dim_campaign, model_artifacts` (only
the first item follows the word FROM), and a string literal containing
`, model_artifacts as (` that registered the name as a trusted CTE. Each of
those is a place where a pattern encoded a guess about SQL's grammar and the
guess was wrong. Adding a fourth pattern would have been a fourth guess, so
the grammar is now sqlglot's problem and the policy is this module's.

**What runs is what this module understood.** `validate_sql` returns SQL
rendered back from the parsed tree, not the model's original text, and that
rendered string is both what executes and what the user is shown. Anything
sqlglot did not understand is therefore absent from what reaches Postgres
rather than passing through unexamined — `LIMIT ALL`, for instance, which
sqlglot drops and which would otherwise have suppressed the row cap.

No I/O of any kind: this module reads a string and returns a string or
raises. That is what makes the whole rule set testable without a database,
and `tests/test_sql_guard.py` asserts the module stays that way.
"""

import unicodedata

import sqlglot
from sqlglot import exp

from adsignal.warehouse.ask_schema import ALLOWED_TABLES

# The dialect every query is parsed and rendered as. The executor runs on
# Postgres, so anything else would be validating a different language.
DIALECT = "postgres"

# The cap every answer is bounded by. A query whose outermost statement has
# no LIMIT gets this one; a query asking for more is refused rather than
# quietly shrunk, because the SQL shown to the user must be the SQL that ran.
MAX_ROWS = 200

# Statement and clause node types that may never appear anywhere in the tree,
# including nested inside a perfectly legal `WITH`. This is what makes a
# writing CTE — `WITH x AS (DELETE … RETURNING *) SELECT * FROM x` — a
# refusal rather than a deletion: the `Delete` node is found wherever it is.
# `Command` is sqlglot's node for syntax it does not model (VACUUM, LISTEN,
# REVOKE), so refusing it also refuses everything this list has not thought
# of. Looked up by name so a future sqlglot release that renames or drops one
# cannot stop the module importing.
_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "AlterTable",
    "Grant",
    "Revoke",
    "TruncateTable",
    "Copy",
    "Merge",
    "Refresh",
    "Command",
    "Set",
    "SetItem",
    "Lock",
    "Transaction",
    "Commit",
    "Rollback",
    "Use",
)
_FORBIDDEN_NODES = tuple(
    node for name in _FORBIDDEN_NODE_NAMES if (node := getattr(exp, name, None)) is not None
)

# Identifiers and function names that may never appear, whatever they name.
# `pg_` is refused as a whole prefix rather than as a list of the interesting
# functions: the list of interesting catalog functions is long and grows with
# each Postgres release, and no answer about campaign performance needs one.
_FORBIDDEN_NAMES = frozenset({"information_schema", "dblink", "dblink_connect", "dblink_exec"})
_FORBIDDEN_PREFIX = "pg_"

# Control, format, surrogate, private-use and unassigned characters. Tab and
# newline are ordinary whitespace; a zero-width joiner inside a keyword has no
# honest use in SQL. Checked before parsing, because the parser would simply
# fail on most of them and "invalid query" is a less useful thing to tell a
# user than "that character is not allowed".
_DISALLOWED_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})
_ALLOWED_CONTROL = frozenset({"\t", "\n", "\r"})


class SqlRejected(ValueError):
    """A generated query the validator refused, and the rule that refused it.

    `reason` is written to be shown to the user: it names what was wrong and
    nothing about the deployment. It never carries a DSN, a driver message
    or a traceback — those are the things a refusal must not leak.
    """

    def __init__(self, reason: str, rule: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.rule = rule


def validate_sql(sql: str) -> str:
    """Return the exact SQL that may execute, or raise `SqlRejected`.

    The returned string is rendered from the parsed tree and carries a
    `LIMIT {MAX_ROWS}` on its outermost statement. It is what runs and what
    the user is shown, so the two cannot differ.
    """
    text = unicodedata.normalize("NFKC", sql).strip()
    if not text:
        raise SqlRejected("the model produced no query", "empty")

    _reject_unsupported_characters(text)
    statement = _single_statement(text)
    _reject_comments(statement)
    query = _require_read_root(statement)
    _reject_forbidden_nodes(query)
    _reject_forbidden_names(query)
    _require_allowlisted_tables(query)
    return _capped(query).sql(dialect=DIALECT)


def _reject_unsupported_characters(text: str) -> None:
    """Refuse control and zero-width characters, which only ever hide things."""
    for character in text:
        if character in _ALLOWED_CONTROL:
            continue
        if unicodedata.category(character) in _DISALLOWED_CATEGORIES:
            raise SqlRejected(
                "the query contains a control or zero-width character",
                "unsupported_characters",
            )


def _single_statement(text: str) -> exp.Expression:
    """Parse `text` and return its one statement, or refuse.

    The parser is what decides where a statement ends, so a semicolon inside
    a string literal is data (as Postgres would read it) while a second real
    statement is a refusal.
    """
    try:
        parsed = sqlglot.parse(text, read=DIALECT)
    except sqlglot.ParseError:
        raise SqlRejected("that is not a query this endpoint can parse", "invalid_query") from None

    statements = [statement for statement in parsed if isinstance(statement, exp.Expression)]
    if not statements:
        raise SqlRejected("the model produced no query", "empty")
    if len(statements) > 1:
        raise SqlRejected("only a single statement may be run", "single_statement")
    return statements[0]


def _reject_comments(statement: exp.Expression) -> None:
    """Refuse SQL comments, in either syntax, anywhere in the tree.

    Read off the parsed nodes rather than by scanning for `--` and `/*`, so a
    string literal that merely contains those characters is data. A comment
    cannot contribute to an answer, and both syntaxes exist here only to
    split a keyword or hide a payload from a reader.
    """
    for node in statement.walk():
        if node.comments:
            raise SqlRejected("SQL comments are not allowed", "no_comments")


def _require_read_root(statement: exp.Expression) -> exp.Query:
    """Return the statement as a query, or refuse it for not being one.

    `exp.Query` is sqlglot's base for everything that reads: a SELECT, a set
    operation (UNION/INTERSECT/EXCEPT) and a parenthesised query. Everything
    else — a write, a DDL statement, or anything sqlglot could only parse as
    a `Command` — is not a query and is refused here. Narrowing the type is
    also what lets `_capped` put a LIMIT on it.
    """
    if not isinstance(statement, exp.Query):
        raise SqlRejected("only SELECT and WITH queries may be run", "select_only")
    if statement.args.get("into") is not None:
        raise SqlRejected("SELECT INTO writes a table and is not allowed", "forbidden_keyword")
    return statement


def _reject_forbidden_nodes(statement: exp.Query) -> None:
    """Refuse writing, administrative and locking constructs wherever they appear."""
    for node in statement.find_all(*_FORBIDDEN_NODES):
        raise SqlRejected(f"{type(node).__name__.upper()} is not allowed here", "forbidden_keyword")


def _reject_forbidden_names(statement: exp.Query) -> None:
    """Refuse catalog identifiers and exfiltration functions by name.

    Every identifier and every function name in the tree is checked, so
    `pg_read_file(...)`, `pg_catalog.pg_authid`, `information_schema.tables`
    and `dblink(...)` are refused wherever they sit — as a table, a schema
    qualifier, a column or a call.
    """
    for node in statement.walk():
        name = _name_of(node)
        if name is None:
            continue
        lowered = name.lower()
        if lowered.startswith(_FORBIDDEN_PREFIX) or lowered in _FORBIDDEN_NAMES:
            raise SqlRejected(
                f"{name} is not allowed; only the documented tables may be read",
                "forbidden_keyword",
            )


def _name_of(node: object) -> str | None:
    """The identifier or function name `node` carries, if it carries one."""
    if isinstance(node, exp.Identifier | exp.Anonymous):
        name = node.name
        return name if isinstance(name, str) and name else None
    return None


def _require_allowlisted_tables(statement: exp.Query) -> None:
    """Refuse any table reference outside the allowlist or the query's own CTEs.

    Every `Table` node is checked, however it was written: quoted, mixed
    case, comma-joined, or nested in a subquery. A schema qualifier is
    refused outright rather than resolved — `public.dim_campaign` is a
    different string from the one that was allowlisted, and the read-only
    role's search path is not this module's to assume. A `Table` with no name
    is a table function (`generate_series(...)`, `dblink(...)`), which is not
    a documented table either.
    """
    defined = {cte.alias.lower() for cte in statement.find_all(exp.CTE) if cte.alias}
    for table in statement.find_all(exp.Table):
        if table.catalog or table.db:
            qualified = ".".join(part for part in (table.catalog, table.db, table.name) if part)
            raise SqlRejected(
                f"{qualified} is not a table this endpoint may read", "table_allowlist"
            )
        name = table.name
        if not name:
            raise SqlRejected(
                "only the documented tables may be read, not table functions",
                "table_allowlist",
            )
        if name.lower() in defined or name.lower() in ALLOWED_TABLES:
            continue
        raise SqlRejected(f"{name} is not a table this endpoint may read", "table_allowlist")


def _capped(statement: exp.Query) -> exp.Query:
    """Refuse any LIMIT above the cap, and put the cap on the outermost query.

    Both halves matter. Refusing an oversized LIMIT keeps the query the user
    is shown honest, since silently shrinking it would display a query that
    never ran. Applying the cap to the outermost statement *regardless of
    inner limits* is what stops `SELECT * FROM (SELECT … LIMIT 5) t JOIN …`
    from returning the whole join: an inner LIMIT bounds a subquery, never
    the result the API has to hold in memory.
    """
    for limit in statement.find_all(exp.Limit):
        value = limit.expression
        if not isinstance(value, exp.Literal) or not value.is_int:
            raise SqlRejected(
                f"LIMIT must be a plain number no greater than {MAX_ROWS}", "row_limit"
            )
        if int(value.name) > MAX_ROWS:
            raise SqlRejected(f"LIMIT must be no greater than {MAX_ROWS}", "row_limit")

    if statement.args.get("limit") is not None:
        return statement
    return statement.limit(MAX_ROWS)
