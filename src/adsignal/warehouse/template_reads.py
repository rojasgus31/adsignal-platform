"""Running `/ask`'s named query templates against Postgres.

`api/queries.py` declares the closed set of templates; this module is the
only place one of them is ever executed, so the layering rule ("api reads
through `warehouse/`, never issues SQL itself") holds for `/ask` exactly as
it does for `/campaigns`, `/anomalies`, `/models` and `/search`.

The reader takes the already-bound pieces of a template — a fixed SQL
literal, its column names, and a tuple of already-validated parameters —
rather than the `QueryTemplate` object itself, so this inner layer never has
to import from the outer `api/` one.
"""

from collections.abc import Sequence
from typing import Any

from adsignal.warehouse.loader import ensure_schema
from adsignal.warehouse.metrics_store import ensure_ml_schema
from adsignal.warehouse.pool import connection


def run_template(
    dsn: str, sql: str, columns: Sequence[str], parameters: Sequence[object]
) -> list[dict[str, Any]]:
    """Run one template's fixed SQL and return its rows as dicts.

    Both `ensure_*_schema` calls run first, for the same reason
    `serving_reads.py` makes them: the templates read `fct_campaign_daily`
    (the mart) and `anomaly_predictions` (the ML outputs), and a database
    that has never had a pipeline run against it has neither table. Creating
    the empty tables first turns that into an empty answer rather than a
    `psycopg.errors.UndefinedTable` 500 on the first visitor of a fresh
    deployment.

    `sql` is always a module-level literal from `api.queries.QUERY_TEMPLATES`
    and `parameters` is always the output of `QueryTemplate.bind`, so no
    request data is ever interpolated into the statement.
    """
    ensure_schema(dsn)
    ensure_ml_schema(dsn)
    with connection(dsn) as conn:
        rows = conn.execute(sql, tuple(parameters)).fetchall()
    return [dict(zip(columns, row, strict=True)) for row in rows]
