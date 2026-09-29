"""The pure validator standing between a generated query and the database.

`/ask` runs SQL a language model wrote. A model instructed to emit only
SELECT is not a security control, so this module assumes the model is
hostile and checks the text it produced. It is one of three controls and the
only one that can explain *why* a query was refused; the other two — a
read-only Postgres role and a read-only transaction with a statement
timeout, both in `warehouse.ask_reads` — hold even if this module is
bypassed entirely.

Deliberately a text validator rather than a parser. A parser that accepts
everything Postgres accepts is a much larger surface than a scanner that
accepts a narrow, boring subset and refuses the rest, and a refusal here
costs a user one rephrase. Where a rule could plausibly reject a legitimate
query — a string literal containing `--`, a schema-qualified table name — it
rejects it anyway.

No I/O of any kind: this module reads its input string and returns a string
or raises. That is what makes the whole rule set testable without a
database, and `tests/test_sql_guard.py` asserts the module stays that way.
"""

import re
import unicodedata

from adsignal.warehouse.ask_schema import ALLOWED_TABLES

# The cap every answer is bounded by. A query with no LIMIT gets this one
# appended; a query asking for more is refused rather than quietly rewritten,
# because the SQL shown to the user must be the SQL that ran.
MAX_ROWS = 200

# Statement keywords and functions that may never appear, whatever they are
# nested inside. `DELETE`/`INSERT`/`UPDATE` cover the writing CTE, which
# opens with a perfectly legal `WITH`. `SET` covers an attempt to undo the
# executor's `SET LOCAL transaction_read_only`. `dblink` and
# `information_schema` are reads, and are refused anyway: one exfiltrates to
# another server, the other enumerates a database the answer never needs.
_FORBIDDEN_WORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "grant",
    "revoke",
    "truncate",
    "copy",
    "call",
    "do",
    "set",
    "vacuum",
    "analyze",
    "merge",
    "refresh",
    "listen",
    "notify",
    "dblink",
    "information_schema",
)
_FORBIDDEN_RE = re.compile(rf"\b({'|'.join(_FORBIDDEN_WORDS)})\b", re.IGNORECASE)

# Every `pg_`-prefixed identifier, not just the named ones. `pg_read_file`,
# `pg_sleep` and `pg_catalog` are the obvious payloads; the prefix as a whole
# is refused because the list of interesting catalog functions is long and
# grows with each Postgres release, and no answer about campaign performance
# needs any of them.
_PG_IDENT_RE = re.compile(r"\bpg_[a-z0-9_]+", re.IGNORECASE)

# An identifier appearing where a table may appear. `FROM (` and `JOIN (`
# introduce a subquery rather than a table, and do not match.
_TABLE_REF_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_.$]*)", re.IGNORECASE)

# CTE names, which are legal table references even though they are not in the
# allowlist. Both forms: the first after `WITH`, and each later one after a
# comma.
_FIRST_CTE_RE = re.compile(r"\bwith\s+(?:recursive\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s+as\b", re.I)
_NEXT_CTE_RE = re.compile(r",\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", re.IGNORECASE)

_LIMIT_RE = re.compile(r"\blimit\b\s*([^\s,)]+)", re.IGNORECASE)

# Control, format, surrogate, private-use and unassigned characters. A tab or
# newline is ordinary whitespace and is allowed; a zero-width joiner inside a
# keyword has no honest use in SQL.
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
    """Return `sql` in the exact form that may be executed, or raise `SqlRejected`.

    The returned string is Unicode-normalised (NFKC), stripped of a trailing
    semicolon, and carries a `LIMIT {MAX_ROWS}` if the query had none. It is
    both what runs and what the user is shown, so the two can never differ.
    """
    text = unicodedata.normalize("NFKC", sql).strip()
    if not text:
        raise SqlRejected("the model produced no query", "empty")

    _reject_unsupported_characters(text)
    _reject_comments(text)
    body = _single_statement(text)
    _require_read_opening(body)
    _reject_forbidden_words(body)
    _require_allowlisted_tables(body)
    return _capped(body)


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


def _reject_comments(text: str) -> None:
    """Refuse SQL comments outright, in either syntax.

    A comment cannot contribute to an answer, and both syntaxes exist here
    only to split a keyword (`DR/**/OP`) or to hide a payload behind a line
    break. Refusing them means no later rule has to reason about them.
    """
    if "--" in text or "/*" in text or "*/" in text:
        raise SqlRejected("SQL comments are not allowed", "no_comments")


def _single_statement(text: str) -> str:
    """Return the one statement in `text`, or refuse if there is more than one.

    One trailing semicolon is punctuation and is dropped. Any other
    semicolon — including one inside a string literal, which this scanner
    does not try to tell apart — is refused.
    """
    body = text[:-1].strip() if text.endswith(";") else text
    if ";" in body:
        raise SqlRejected("only a single statement may be run", "single_statement")
    return body


def _require_read_opening(body: str) -> None:
    """Refuse anything not opening with SELECT or WITH."""
    if not re.match(r"^(select|with)\b", body, re.IGNORECASE):
        raise SqlRejected("only SELECT and WITH queries may be run", "select_only")


def _reject_forbidden_words(body: str) -> None:
    """Refuse writing, administrative and catalog constructs wherever they appear."""
    found = _FORBIDDEN_RE.search(body)
    if found is not None:
        raise SqlRejected(f"{found.group(1).upper()} is not allowed", "forbidden_keyword")
    catalog = _PG_IDENT_RE.search(body)
    if catalog is not None:
        raise SqlRejected(
            f"{catalog.group(0)} is not allowed; only the documented tables may be read",
            "forbidden_keyword",
        )


def _require_allowlisted_tables(body: str) -> None:
    """Refuse any table reference outside the allowlist or the query's own CTEs."""
    defined = {match.group(1).lower() for match in _FIRST_CTE_RE.finditer(body)}
    defined |= {match.group(1).lower() for match in _NEXT_CTE_RE.finditer(body)}
    for match in _TABLE_REF_RE.finditer(body):
        name = match.group(1).lower()
        if name in defined or name in ALLOWED_TABLES:
            continue
        raise SqlRejected(f"{name} is not a table this endpoint may read", "table_allowlist")


def _capped(body: str) -> str:
    """Append the row cap, or refuse a LIMIT that is missing, larger or not a number."""
    limits = _LIMIT_RE.findall(body)
    if not limits:
        return f"{body} LIMIT {MAX_ROWS}"
    for raw in limits:
        if not raw.isdigit():
            raise SqlRejected(
                f"LIMIT must be a plain number no greater than {MAX_ROWS}", "row_limit"
            )
        if int(raw) > MAX_ROWS:
            raise SqlRejected(f"LIMIT must be no greater than {MAX_ROWS}", "row_limit")
    return body
