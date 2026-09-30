"""The facts one campaign's written summary is built from.

A separate module from `campaign_reads` because it answers a different
question. That one serves rows for charting; this one aggregates a campaign's
whole run into the closed set of numbers a prompt is allowed to see.

The layering rule holds as everywhere else: this is the only place that knows
SQL for this feature, and `api.routes.summary` calls it rather than issuing a
query of its own.
"""

from adsignal.narrative.campaign_summary import CampaignFacts
from adsignal.warehouse.loader import ensure_schema as ensure_mart_schema
from adsignal.warehouse.metrics_store import ensure_ml_schema
from adsignal.warehouse.pool import connection

# Totals plus the run's bounds, from the mart alone. `is_anomaly` here is
# ground truth (a planted day), which is a different number from the
# detector's flags below, and the two are read separately so nothing
# downstream can confuse them.
#
# GROUP BY names the underlying columns, never the SELECT aliases:
# `fct_campaign_daily` has its own `channel`, so `GROUP BY channel` is
# ambiguous and Postgres refuses it.
_TOTALS_SQL = """
SELECT COALESCE(NULLIF(dim.campaign_name, ''), daily.campaign_id) AS campaign_name,
       COALESCE(NULLIF(dim.channel, ''), 'unknown') AS channel,
       MIN(daily.event_date) AS first_day,
       MAX(daily.event_date) AS last_day,
       COUNT(*) AS days,
       COALESCE(SUM(daily.spend), 0) AS spend,
       COALESCE(SUM(daily.conversions), 0) AS conversions,
       COALESCE(SUM(daily.revenue), 0) AS revenue,
       COUNT(*) FILTER (WHERE daily.is_anomaly) AS planted_days
FROM fct_campaign_daily AS daily
LEFT JOIN dim_campaign AS dim ON dim.campaign_id = daily.campaign_id
WHERE daily.campaign_id = %s
GROUP BY dim.campaign_name, dim.channel, daily.campaign_id
"""

# What the detector and triage models said about this campaign. Separate
# query rather than a join: `anomaly_predictions` has at most one row per
# campaign-day, and joining it to the daily rows before aggregating would
# multiply nothing here but invites exactly that bug on the next edit.
_FLAGS_SQL = """
SELECT COUNT(*) AS flagged_days,
       COUNT(*) FILTER (WHERE escalate) AS escalated_days,
       MAX(event_date) AS last_flagged
FROM anomaly_predictions
WHERE campaign_id = %s
"""

# The causes this campaign's flagged days were attributed to, commonest
# first. Limited because a prompt gets the headline, not an inventory.
_CAUSES_SQL = """
SELECT cause, COUNT(*) AS n
FROM anomaly_predictions
WHERE campaign_id = %s
GROUP BY cause
ORDER BY n DESC, cause ASC
LIMIT 3
"""


def read_campaign_facts(dsn: str, campaign_id: str) -> CampaignFacts | None:
    """Aggregate one campaign's run, or `None` when it has no daily rows.

    `None` (not an exception) for an unknown campaign, so the router decides
    the status code. Both schemas are ensured first: a database that has
    never run the pipeline has neither table, and a missing-relation error
    would surface as a 500 for what is really an empty warehouse.
    """
    ensure_mart_schema(dsn)
    ensure_ml_schema(dsn)
    with connection(dsn) as conn:
        totals = conn.execute(_TOTALS_SQL, (campaign_id,)).fetchone()
        if totals is None:
            return None
        flags = conn.execute(_FLAGS_SQL, (campaign_id,)).fetchone()
        causes = conn.execute(_CAUSES_SQL, (campaign_id,)).fetchall()

    name, channel, first_day, last_day, days, spend, conversions, revenue, planted = totals
    # `fetchone` on an aggregate with no GROUP BY always returns a row, but
    # COUNT over no rows is 0 and MAX over no rows is NULL.
    flagged_days, escalated_days, last_flagged = flags if flags is not None else (0, 0, None)

    return CampaignFacts(
        campaign_id=campaign_id,
        campaign_name=str(name),
        channel=str(channel),
        first_day=first_day,
        last_day=last_day,
        days=int(days),
        spend=float(spend),
        conversions=int(conversions),
        revenue=float(revenue),
        planted_days=int(planted),
        flagged_days=int(flagged_days or 0),
        escalated_days=int(escalated_days or 0),
        last_flagged=last_flagged,
        top_causes=tuple(str(cause) for cause, _ in causes),
    )
