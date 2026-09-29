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

**And rendering is checked, because rendering was itself a hole.** That
reasoning held in one direction and not the other: sqlglot can *emit*
something it models differently from Postgres. Review found
`SELECT E'\\' AS a, ', b FROM model_artifacts --' AS c FROM dim_campaign`
accepted, because sqlglot rendered the escape string as `e'\\'`, where
Postgres reads `\\'` as an escaped quote rather than a terminator. The
literal's boundary moved between what was validated and what would have run,
and the rendered query re-parses to `FROM model_artifacts` with the row cap
commented out — three rules defeated by one payload.

So every query is validated **twice**: once as written, and again as
rendered, with the two required to agree on their real table set, their
comment absence and their effective cap, and the rendering required to be a
fixed point. A validator whose own output re-parses to something else has not
finished its job.

No I/O of any kind: this module reads a string and returns a string or
raises. That is what makes the whole rule set testable without a database,
and `tests/test_sql_guard.py` asserts the module stays that way.
"""

import unicodedata
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

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
# Both spellings of a row bound. `Fetch` is `FETCH FIRST n ROWS ONLY`, which
# is LIMIT in standard SQL clothing and was missed by the first parser
# version.
_BOUND_NODES = tuple(
    node for name in ("Limit", "Fetch") if (node := getattr(exp, name, None)) is not None
)

# Function names sqlglot does not recognise, which are nonetheless allowed.
#
# Every entry is a promise about the named function: it cannot take SQL as an
# argument, it cannot read outside the allowlisted tables, and it cannot
# return an unbounded value. The list started empty on the principle that
# each promise should be made deliberately and one at a time. Review then
# measured what that cost: **eighteen** legitimate functions were refused,
# roughly one in five of what a language model emits for an ordinary
# question, and they clustered exactly where questions cluster — dates and
# JSON. "Empty by default" is only honest if entries get added as they
# surface, and these surfaced.
#
# Each was checked against all three promises; the per-function reasoning is
# in the task report. `setseed` is deliberately absent: it mutates session
# state and answers no question about campaign performance.
#
# The third promise is the weakest and is worth naming here rather than
# leaving implied. `jsonb_agg` and `row_to_json` can build one value out of
# every row in a table — and so can `array_agg` and `string_agg`, which
# sqlglot models and which this list therefore never had a say over. What
# actually bounds the size of an answer is `ask_reads.MAX_RESULT_BYTES`, not
# this list.
_ALLOWED_UNKNOWN_FUNCTIONS: frozenset[str] = frozenset(
    {
        # Dates and times: scalar in, scalar out.
        "age",
        "make_date",
        "isfinite",
        "clock_timestamp",
        "statement_timestamp",
        "timezone",
        # Scalar helpers over values the query already has.
        "num_nonnulls",
        "cardinality",
        "quote_literal",
        "regexp_matches",
        "regexp_split_to_array",
        # Aggregates over allowlisted rows.
        "every",
        # JSON construction. Builds a value from columns already selected;
        # reaches nothing the query could not already read.
        "to_json",
        "row_to_json",
        "json_build_object",
        "jsonb_build_object",
        "jsonb_agg",
    }
)

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


@dataclass(frozen=True, slots=True)
class _Shape:
    """What a query does, reduced to the things the round-trip must preserve.

    The real tables it reads (CTE names resolved away) and the row bound it
    will actually be subject to. Two parses of what should be the same query
    must agree on both.

    Comment absence is deliberately *not* a field here. It was, and the field
    was a lie: `_reject_comments` raises, so it could only ever hold `False`,
    and the round-trip clause comparing it was dead code asserting a live
    guarantee. What actually protects the rendered form is that `_analyse`
    runs `_reject_comments` on it too, and raises.
    """

    tables: frozenset[str]
    row_bound: int | None


def validate_sql(sql: str) -> str:
    """Return the exact SQL that may execute, or raise `SqlRejected`.

    The returned string is rendered from the parsed tree and carries a row
    bound of at most `MAX_ROWS` on its outermost query. It is what runs and
    what the user is shown, so the two cannot differ — and it has been
    re-parsed and re-validated in that exact form, so what Postgres will read
    is what this module checked.
    """
    text = unicodedata.normalize("NFKC", sql).strip()
    if not text:
        raise SqlRejected("the model produced no query", "empty")

    _reject_unsupported_characters(text)
    query, shape = _analyse(text)
    rendered = _capped(query).sql(dialect=DIALECT)
    _require_round_trip(rendered, shape)
    return rendered


def _analyse(text: str) -> tuple[exp.Query, _Shape]:
    """Parse `text`, enforce every rule on it, and describe what it does.

    Run twice per query — once on the model's text and once on this module's
    own rendering of it — so the second run is a real re-validation and not a
    cheaper summary of the first.
    """
    statement = _single_statement(text)
    _reject_comments(statement)
    query = _requoted(_require_read_root(statement))
    _reject_forbidden_nodes(query)
    _reject_forbidden_names(query)
    _require_known_functions(query)
    tables = _require_allowlisted_tables(query)
    bound = _row_bound(query)
    return query, _Shape(tables=tables, row_bound=bound)


def _require_round_trip(rendered: str, original: _Shape) -> None:
    """Re-validate this module's own output and require it to mean the same thing.

    Three ways to fail, any one of which means the rendering says something
    to Postgres that the validated tree did not say: the rendered query reads
    a different set of real tables, it is not bounded by the cap after all,
    or rendering it again produces different text (so there is no stable
    string to reason about at all). The refusal deliberately does not explain
    which, because the query that produced it is not one to iterate against.

    Every other rule is enforced on the rendered form too, by `_analyse`
    itself raising — comment absence among them, which is why there is no
    comment field on `_Shape` to compare.
    """
    try:
        rendered_query, rendered_shape = _analyse(rendered)
    except SqlRejected as rejection:
        raise SqlRejected(
            "that query does not survive being read back, so it will not be run",
            "round_trip",
        ) from rejection

    bound = rendered_shape.row_bound
    if (
        rendered_shape.tables != original.tables
        or bound is None
        or bound > MAX_ROWS
        or rendered_query.sql(dialect=DIALECT) != rendered
    ):
        raise SqlRejected(
            "that query does not survive being read back, so it will not be run",
            "round_trip",
        )


def _requoted(statement: exp.Query) -> exp.Query:
    """Restore string literals sqlglot parsed into bare `Var` nodes.

    `date_bin('1 week', event_date, '2024-01-01')` parses its origin as a
    `Var` named `2024-01-01` and renders it unquoted, so the rendering
    re-parses as the arithmetic `2024 - 01 - 01`. The round-trip check
    refused it, correctly — but the query was ordinary and the renderer was
    at fault, so a refusal was the wrong answer to give the user.

    **The two forms are indistinguishable after parsing, and this function
    chooses one.** sqlglot parses `date_bin('1 week', ts, 'event_date')` and
    `date_bin('1 week', ts, event_date)` — a string literal and a bare column
    reference — into the *same* node: `Var('EVENT_DATE')` under `unit`, with
    a `DateBin` parent. Nothing in the tree distinguishes them, so no rule
    here can get both right. Round 4 treated the position as a column and got
    the literal case wrong (the literal silently became a column reference and
    Postgres returned a row). This version treats the position as a literal,
    which gets the literal case right and makes a bare-column origin a
    refusal: it renders as `'EVENT_DATE'`, and Postgres rejects the cast.

    That is the deliberate direction. A refusal is a worse answer than a
    correct one and a far better outcome than a wrong one, and the wrong
    answer in the other direction was silent — round-trip and fixed-point
    cannot see it, because the rendering is stable and the table set is
    unchanged.

    The other half of the job is unambiguous and is not a trade: a `Var`
    whose text is not a valid identifier (`'2024-01-01'`) can only have come
    from a string literal, and rendering it bare produced the arithmetic
    `2024 - 01 - 01`, which the round-trip check refused. Quoting it is a
    plain repair of a rendering defect.

    **Case is not preserved and cannot be.** sqlglot upper-cases the token
    while tokenising, so the original text is gone before this runs. For a
    date or timestamp origin that changes nothing.

    `DateBin` is not the only node sqlglot renders a `Var` bare from —
    `Extract` does too, reachable as `date_part('week', d)` → `EXTRACT(WEEK
    FROM d)`. That one is left alone and is meaning-preserving: Postgres's
    EXTRACT field is a unit name, not a column reference, so a bare word
    there is what was meant. Only `DateBin` is rewritten.

    Applied inside `_analyse`, so both passes normalise identically —
    otherwise the rendering would be re-parsed back into a `Var` and the
    fixed-point clause would refuse the repair itself.
    """
    for var in statement.find_all(exp.Var):
        name = var.name
        if not isinstance(name, str) or not name:
            continue
        if isinstance(var.parent, exp.DateBin) or not name.isidentifier():
            var.replace(exp.Literal.string(name))
    return statement


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
    except sqlglot.errors.SqlglotError:
        # `SqlglotError`, not `ParseError`. `TokenError` is a *sibling* of
        # `ParseError` - both descend straight from `SqlglotError` - so
        # catching the narrower type let an unterminated string literal
        # escape `validate_sql` and become a 500 at the route. That is
        # exactly what a model emits when it hits its token limit, which
        # makes it one of the likelier inputs this endpoint sees.
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
        for name in _names_of(node):
            lowered = name.lower()
            if lowered.startswith(_FORBIDDEN_PREFIX) or lowered in _FORBIDDEN_NAMES:
                raise SqlRejected(
                    f"{name} is not allowed; only the documented tables may be read",
                    "forbidden_keyword",
                )


def _require_known_functions(statement: exp.Query) -> None:
    """Refuse every function sqlglot does not recognise as a standard one.

    **This is an allow-list, and the shape matters more than the list.** The
    rule used to be a deny-list — `pg_*`, `information_schema`, `dblink` —
    and review walked straight past it with `query_to_xml('SELECT * FROM
    model_artifacts', …)`, which matches none of those patterns, takes SQL
    as a *string argument*, and returns an entire table as one value. The
    validator saw `FROM dim_campaign`, appended `LIMIT 200`, and Postgres
    executed a second query the validator never read. The reviewer's payload
    came back as a single 8 MB value, so the row cap and the bounded fetch
    were defeated along with the allowlist.

    A deny-list has to name a function before it can refuse it, and the set
    of things to name is not knowable: `table_to_xml`, `schema_to_xml`,
    `database_to_xml` and `query_to_xmlschema` are the same hazard in four
    more spellings, and the next extension installed on the server adds more.

    sqlglot parses a function it models into a typed node (`exp.Sum`,
    `exp.DateTrunc`, `exp.Rank`) and everything else into `exp.Anonymous`.
    Refusing `Anonymous` therefore says: *if this is not a standard function
    the parser recognises, it does not run.* The permitted set is bounded by
    sqlglot's own function registry rather than by what anyone here thought
    of, and every future `*_to_xml`, catalog helper and extension function is
    refused without being named.

    The cost is honest and small: a legitimate Postgres-specific function
    that sqlglot does not model — `age(...)` is one — is refused, and the
    question has to be rephrased. Refusing a working query is a worse answer;
    running one nobody validated is a worse outcome.
    """
    for call in statement.find_all(exp.Anonymous):
        name = call.name
        if isinstance(name, str) and name.lower() in _ALLOWED_UNKNOWN_FUNCTIONS:
            continue
        raise SqlRejected(f"{name}() is not a function this endpoint runs", "forbidden_function")


def _names_of(node: object) -> tuple[str, ...]:
    """Every identifier-ish name `node` carries.

    Operators are the reason this returns a tuple rather than one name.
    `OPERATOR(pg_catalog.||)` keeps its qualifier in a plain string argument
    rather than in an `Identifier`, so scanning identifiers alone let a
    `pg_catalog`-qualified operator through. No core operator takes SQL or
    touches the filesystem, so nothing was exploitable — but the README says
    no `pg_*` identifier reaches the database, and that sentence should be
    true rather than nearly true.
    """
    if isinstance(node, exp.Identifier | exp.Anonymous):
        name = node.name
        return (name,) if isinstance(name, str) and name else ()
    if isinstance(node, exp.Operator):
        operator = node.args.get("operator")
        if isinstance(operator, str) and operator:
            return tuple(part for part in operator.split(".") if part)
    return ()


def _require_allowlisted_tables(statement: exp.Query) -> frozenset[str]:
    """Refuse any real table outside the allowlist, and return the ones it reads.

    **Scope-aware, because collecting CTE names tree-wide was wrong.** A
    `WITH` nested inside a sub-select declares a name only for that
    sub-select, but a tree-wide collection let it launder the same name for
    the outer query:

        SELECT * FROM model_artifacts WHERE 1 IN
          (WITH model_artifacts AS (SELECT 1 AS x) SELECT x FROM model_artifacts)

    sqlglot's scope machinery resolves each reference in the scope that
    declares it, so the outer `model_artifacts` is a real table (refused)
    while the inner one is a CTE reference (allowed). That is the same class
    of error as the string-literal CTE bug, made with valid syntax.

    Anything the scope walk does not classify as a source is checked too, and
    is allowed only if it names a CTE the query really declares — so a
    reference the analyser did not reach fails closed rather than silently.
    """
    declared = {cte.alias.lower() for cte in statement.find_all(exp.CTE) if cte.alias}
    real_sources = _real_table_sources(statement)

    tables: set[str] = set()
    for table in statement.find_all(exp.Table):
        if id(table) in real_sources:
            tables.add(_checked_table_name(table))
            continue
        # Not a real source: a CTE reference, or something scope resolved
        # away. It may only be the former.
        if table.catalog or table.db or table.name.lower() not in declared:
            tables.add(_checked_table_name(table))
    return frozenset(tables)


def _real_table_sources(statement: exp.Query) -> set[int]:
    """The `id()` of every `Table` node scope analysis resolves to a real table.

    Falls back to "every table node is real" if sqlglot cannot build a scope
    for this query. That is the fail-closed direction: it removes the CTE
    exemption entirely rather than granting it blindly.
    """
    try:
        root = build_scope(statement)
    except Exception:  # noqa: BLE001 - any analyser failure falls back to strict
        root = None
    if root is None:
        return {id(table) for table in statement.find_all(exp.Table)}

    sources: set[int] = set()
    for scope in root.traverse():
        for source in scope.sources.values():
            if isinstance(source, exp.Table):
                sources.add(id(source))
    return sources


def _checked_table_name(table: exp.Table) -> str:
    """The table's bare name, or a refusal if it is not one this endpoint reads.

    A schema qualifier is refused rather than resolved — `public.dim_campaign`
    is a different string from the one that was allowlisted, and the
    read-only role's search path is not this module's to assume. A `Table`
    with no name is a table function (`generate_series(...)`), which is not a
    documented table either.
    """
    if table.catalog or table.db:
        qualified = ".".join(part for part in (table.catalog, table.db, table.name) if part)
        raise SqlRejected(f"{qualified} is not a table this endpoint may read", "table_allowlist")
    name = table.name
    if not name:
        raise SqlRejected(
            "only the documented tables may be read, not table functions", "table_allowlist"
        )
    if name.lower() not in ALLOWED_TABLES:
        raise SqlRejected(f"{name} is not a table this endpoint may read", "table_allowlist")
    return name.lower()


def _row_bound(statement: exp.Query) -> int | None:
    """The row bound the outermost query carries, refusing any that is too large.

    Both spellings count. `FETCH FIRST n ROWS ONLY` is LIMIT in standard SQL
    clothing, and the first parser version knew only `Limit`: an oversized
    `FETCH FIRST` was accepted and rendered uncapped, turning a rule that had
    been failing closed (a syntax error) into one that failed open. Every
    bound in the tree is range-checked, inner ones included; the value
    returned is the outermost one, which is what actually bounds the answer.
    """
    for node in statement.find_all(*_BOUND_NODES):
        _bound_value(node)
    outer = statement.args.get("limit") or statement.args.get("fetch")
    return None if outer is None else _bound_value(outer)


def _bound_value(node: exp.Expression) -> int:
    """The integer a LIMIT or FETCH carries, or a refusal.

    The two spell their count differently: `Limit` keeps it as the node's
    expression, `Fetch` under a `count` argument. sqlglot stores a `Fetch`
    under the same `limit` key as a `Limit`, so reading only the expression
    made every `FETCH FIRST n ROWS ONLY` look like a non-numeric bound.
    """
    if isinstance(node, exp.Fetch):
        options = node.args.get("limit_options")
        if options is not None and options.args.get("percent"):
            # `FETCH FIRST 50 PERCENT ROWS ONLY` is not a bound of 50 rows.
            # It used to be read as 50 and rendered with PERCENT intact, so
            # Postgres rejected the result and it surfaced as a parse
            # failure - fail-closed by accident rather than by rule.
            raise SqlRejected(
                f"a row limit must be a plain number of rows, no greater than {MAX_ROWS}",
                "row_limit",
            )
    value = node.args.get("count") if isinstance(node, exp.Fetch) else node.expression
    if isinstance(value, str):
        # Some sqlglot versions keep the count as a plain string.
        value = exp.Literal.number(value) if value.isdigit() else None
    if not isinstance(value, exp.Literal) or not value.is_int:
        raise SqlRejected(
            f"a row limit must be a plain number no greater than {MAX_ROWS}", "row_limit"
        )
    count = int(value.name)
    if count > MAX_ROWS:
        raise SqlRejected(f"a row limit must be no greater than {MAX_ROWS}", "row_limit")
    return count


def _capped(statement: exp.Query) -> exp.Query:
    """Put the row cap on the outermost query when it does not already have one.

    Applied *regardless of inner bounds*: an inner LIMIT bounds a subquery,
    never the result the API has to hold in memory, so
    `SELECT * FROM (SELECT … LIMIT 5) t JOIN …` still gets the outer cap.
    An oversized bound is refused by `_row_bound` rather than shrunk here,
    because silently shrinking it would show the user a query that never ran.
    """
    if statement.args.get("limit") is not None or statement.args.get("fetch") is not None:
        return statement
    return statement.limit(MAX_ROWS)
