"""The facts a whole-warehouse summary is built from.

The portfolio-wide sibling of `campaign_summary_reads`. Same shape, same
layering rule: the SQL lives here and `api.routes.summary` calls it.
"""

from adsignal.narrative.portfolio_summary import PortfolioFacts
from adsignal.warehouse.loader import ensure_schema as ensure_mart_schema
from adsignal.warehouse.metrics_store import ensure_ml_schema
from adsignal.warehouse.pool import connection

# `days` counts DISTINCT event_date, not rows: the mart holds one row per
# campaign-day, so COUNT(*) would report tens of thousands of "days" for a
# two-year window. Channel comes from the dimension because the fact table's
# own copy is per row.
_TOTALS_SQL = """
SELECT COUNT(DISTINCT daily.campaign_id) AS campaigns,
       COUNT(DISTINCT dim.channel) AS channels,
       MIN(daily.event_date) AS first_day,
       MAX(daily.event_date) AS last_day,
       COUNT(DISTINCT daily.event_date) AS days,
       COALESCE(SUM(daily.spend), 0) AS spend,
       COALESCE(SUM(daily.conversions), 0) AS conversions,
       COALESCE(SUM(daily.revenue), 0) AS revenue,
       COUNT(*) FILTER (WHERE daily.is_anomaly) AS planted_days
FROM fct_campaign_daily AS daily
LEFT JOIN dim_campaign AS dim ON dim.campaign_id = daily.campaign_id
"""

_FLAGS_SQL = """
SELECT COUNT(*) AS flagged_days,
       COUNT(*) FILTER (WHERE escalate) AS escalated_days
FROM anomaly_predictions
"""

_CAUSES_SQL = """
SELECT cause, COUNT(*) AS n
FROM anomaly_predictions
GROUP BY cause
ORDER BY n DESC, cause ASC
LIMIT 3
"""

# The single worst campaign by flag count, named rather than identified: a
# summary that says `cmp_0002_004` tells an executive nothing. The LEFT JOIN
# falls back to the id because nothing forces a predicted campaign to have a
# dimension row.
#
# GROUP BY names both underlying columns, never the output alias. Postgres
# accepts an alias there but then rejects the `predictions.campaign_id` inside
# the COALESCE as ungrouped, which is what this query first did. Grouping by
# the id as well is free: it is unique per campaign.
_WORST_SQL = """
SELECT COALESCE(NULLIF(dim.campaign_name, ''), predictions.campaign_id) AS campaign_name,
       COUNT(*) AS n
FROM anomaly_predictions AS predictions
LEFT JOIN dim_campaign AS dim ON dim.campaign_id = predictions.campaign_id
GROUP BY predictions.campaign_id, dim.campaign_name
ORDER BY n DESC, campaign_name ASC
LIMIT 1
"""


def read_portfolio_facts(dsn: str) -> PortfolioFacts | None:
    """Aggregate every campaign, or `None` when the mart holds no rows.

    `None` (not an exception) on an empty or never-loaded warehouse, so the
    router decides the status code rather than surfacing a 500 for what is
    really a database nothing has run against.
    """
    ensure_mart_schema(dsn)
    ensure_ml_schema(dsn)
    with connection(dsn) as conn:
        totals = conn.execute(_TOTALS_SQL).fetchone()
        flags = conn.execute(_FLAGS_SQL).fetchone()
        causes = conn.execute(_CAUSES_SQL).fetchall()
        worst = conn.execute(_WORST_SQL).fetchone()

    if totals is None:
        return None
    campaigns, channels, first_day, last_day, days, spend, conversions, revenue, planted = totals
    # An aggregate with no GROUP BY always returns a row, but over zero rows
    # MIN and MAX are NULL. That is the empty warehouse, not a summary of one.
    if first_day is None or last_day is None:
        return None

    flagged_days, escalated_days = flags if flags is not None else (0, 0)
    worst_name, worst_flags = worst if worst is not None else ("", 0)

    return PortfolioFacts(
        campaign_count=int(campaigns),
        channel_count=int(channels),
        first_day=first_day,
        last_day=last_day,
        days=int(days),
        spend=float(spend),
        conversions=int(conversions),
        revenue=float(revenue),
        planted_days=int(planted),
        flagged_days=int(flagged_days or 0),
        escalated_days=int(escalated_days or 0),
        top_causes=tuple(str(cause) for cause, _ in causes),
        worst_campaign=str(worst_name or ""),
        worst_campaign_flags=int(worst_flags or 0),
    )
