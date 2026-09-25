"""Named, parameterized query templates for `POST /ask`.

The spec calls `/ask` "natural language to SQL". This module deliberately
does **not** generate SQL: a model — or, with no key configured, a keyword
matcher in `routes/ask.py` — chooses among a fixed, closed set of templates
declared here by name, and each template's own parameters are validated
before binding. Every `sql` string below is a fixed literal with `%s`
placeholders; nothing in this module ever builds a query string out of
request data — not the template name, not a parameter, not a column, not an
ORDER BY direction. A model that cannot emit SQL cannot emit an injection.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import psycopg


@dataclass(frozen=True, slots=True)
class IntParam:
    """An integer parameter, validated to fall within [minimum, maximum].

    A missing value falls back to `default` rather than failing — the caller
    (a keyword matcher, or a model with no notion of "limit") is not expected
    to always supply one.
    """

    minimum: int
    maximum: int
    default: int

    def validate(self, value: object) -> int:
        if value is None:
            return self.default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
        if not (self.minimum <= value <= self.maximum):
            raise ValueError(f"must be between {self.minimum} and {self.maximum}")
        return value


@dataclass(frozen=True, slots=True)
class QueryTemplate:
    """One named, parameterized query.

    `sql` is always a fixed literal with `%s` placeholders. `columns` names
    the result columns in the order `sql`'s SELECT list produces them, used to
    turn each result row into a dict. `params` declares, by name, every
    parameter this template accepts and how to validate it — the only path by
    which a request's data ever reaches the query.
    """

    name: str
    description: str
    sql: str
    columns: tuple[str, ...]
    params: Mapping[str, IntParam] = field(default_factory=dict)

    def bind(self, parameters: Mapping[str, object]) -> tuple[object, ...]:
        """Validate `parameters` against this template's declared params.

        Returns already-validated positional values, in declaration order,
        ready for `%s` binding. Raises `ValueError` if any value fails its
        parameter's own validation — the caller turns that into a 4xx before
        any query ever runs, never into a query.
        """
        return tuple(spec.validate(parameters.get(name)) for name, spec in self.params.items())


QUERY_TEMPLATES: dict[str, QueryTemplate] = {
    "anomaly_count": QueryTemplate(
        name="anomaly_count",
        description="How many anomalies have been detected in total.",
        sql="SELECT COUNT(*) AS count FROM anomaly_predictions",
        columns=("count",),
    ),
    "top_causes": QueryTemplate(
        name="top_causes",
        description="The most common anomaly causes, most frequent first.",
        sql=(
            "SELECT cause, COUNT(*) AS occurrences FROM anomaly_predictions "
            "GROUP BY cause ORDER BY occurrences DESC LIMIT %s"
        ),
        columns=("cause", "occurrences"),
        params={"limit": IntParam(minimum=1, maximum=20, default=5)},
    ),
    "worst_cpa_campaigns": QueryTemplate(
        name="worst_cpa_campaigns",
        description="The campaigns with the highest cost per acquisition.",
        sql=(
            "SELECT campaign_id, cpa FROM fct_campaign_daily "
            "WHERE cpa IS NOT NULL ORDER BY cpa DESC LIMIT %s"
        ),
        columns=("campaign_id", "cpa"),
        params={"limit": IntParam(minimum=1, maximum=20, default=5)},
    ),
    "spend_by_channel": QueryTemplate(
        name="spend_by_channel",
        description="Total spend grouped by channel, highest spend first.",
        sql=(
            "SELECT channel, SUM(spend) AS total_spend FROM fct_campaign_daily "
            "GROUP BY channel ORDER BY total_spend DESC"
        ),
        columns=("channel", "total_spend"),
    ),
}


def run_template(
    dsn: str, template: QueryTemplate, parameters: Mapping[str, object]
) -> list[dict[str, Any]]:
    """Bind `parameters` against `template`, run it, and return rows as dicts.

    `template.bind` is the only place request data reaches the query, and it
    returns a tuple of already-validated values — never a string that could
    be concatenated into `sql`.
    """
    bound = template.bind(parameters)
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(template.sql, bound).fetchall()
    return [dict(zip(template.columns, row, strict=True)) for row in rows]
