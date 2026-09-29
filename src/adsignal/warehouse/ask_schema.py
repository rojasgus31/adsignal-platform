"""What `/ask` is allowed to read, and how that is described to a model.

Two pieces of data, deliberately in one place so they cannot drift apart: the
allowlist the validator enforces and the read-only role is granted, and the
prose description of those same tables that goes into the model's prompt.
Describing a table the validator would reject only produces queries that get
refused.

`model_artifacts` and `entity_embeddings` are absent on purpose. One holds
pickled estimator bytes, the other raw float vectors; neither can contribute
to a natural-language answer about campaign performance, and either could
return megabytes of binary in a single row. They are excluded here, excluded
from the grant in `ask_reads.ensure_readonly_role`, and explicitly revoked
there as well.

Pure. This module opens nothing and reads no environment, so it is safe for
the API layer to import and pass to a model provider. Its only imports are
`dataclasses`, `datetime` and `re`, all from the standard library.

The date range is **not** a constant here. `config.start_date`/`end_date` say
which days were *requested* of the generator; what the warehouse actually
holds is a `MIN`/`MAX` over `fct_campaign_daily`, which `warehouse.ask_reads`
reads and hands back as a `DateCoverage`. `describe_schema` renders it.
"""

import re
from dataclasses import dataclass
from datetime import date

# The only tables a generated query may reference. The validator
# (`warehouse.sql_guard`) rejects any other identifier after FROM or JOIN,
# and the read-only role holds SELECT on exactly these and nothing else.
ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        "fct_campaign_daily",
        "dim_campaign",
        "anomaly_predictions",
        "model_metrics",
        "pipeline_runs",
        "narrative_runs",
        "drift_reports",
    }
)

# The fact table whose `event_date` defines what the corpus covers, and the
# name of that column. Every other allowlisted table is derived from the same
# days, so this one range describes the whole warehouse.
COVERAGE_TABLE = "fct_campaign_daily"
DATE_COLUMN = "event_date"

# The schema as the model sees it. It carries table and column names and
# nothing else: no DSN, no host, no credential, no row of data. Kept in the
# same order a reader would want them — facts first, then dimensions, then
# the operational tables.
SCHEMA_DESCRIPTION = """\
fct_campaign_daily(event_date DATE, campaign_id TEXT, advertiser_id TEXT,
  channel TEXT, impressions BIGINT, clicks BIGINT, conversions BIGINT,
  spend DOUBLE PRECISION, revenue DOUBLE PRECISION, ctr DOUBLE PRECISION,
  cpc DOUBLE PRECISION, cpm DOUBLE PRECISION, cpa DOUBLE PRECISION,
  roas DOUBLE PRECISION, anomaly_cause TEXT, anomaly_severity INTEGER,
  is_anomaly BOOLEAN)
  -- one row per campaign per day; the core performance fact table.

dim_campaign(campaign_id TEXT, campaign_name TEXT, advertiser_id TEXT,
  advertiser_name TEXT, channel TEXT)
  -- human-readable names for a campaign and its advertiser.

anomaly_predictions(event_date DATE, campaign_id TEXT, run_id TEXT,
  cause TEXT, confidence DOUBLE PRECISION, severity INTEGER,
  escalate BOOLEAN, anomaly_score DOUBLE PRECISION)
  -- one row per campaign-day the detector flagged, with its triaged cause.

model_metrics(run_id TEXT, trained_at TIMESTAMPTZ,
  triage_accuracy DOUBLE PRECISION, triage_macro_f1 DOUBLE PRECISION,
  triage_n_samples INTEGER, detector_precision DOUBLE PRECISION,
  detector_recall DOUBLE PRECISION, detector_f1 DOUBLE PRECISION,
  detector_roc_auc DOUBLE PRECISION, artifact_uri TEXT)
  -- one row per training run.

drift_reports(run_id TEXT, measured_at TIMESTAMPTZ, feature TEXT,
  psi DOUBLE PRECISION, ks_statistic DOUBLE PRECISION,
  ks_pvalue DOUBLE PRECISION, drifted BOOLEAN)
  -- per-feature drift statistics, one row per feature per run.

pipeline_runs(run_id TEXT, asset TEXT, started_at TIMESTAMPTZ,
  duration_seconds DOUBLE PRECISION, rows_written INTEGER, status TEXT,
  detail TEXT)
  -- one row per materialised pipeline asset.

narrative_runs(as_of DATE, text TEXT, provider TEXT,
  latency_ms DOUBLE PRECISION, generated_at TIMESTAMPTZ)
  -- the generated daily summaries and which provider wrote each.
"""


@dataclass(frozen=True, slots=True)
class DateCoverage:
    """The first and last `event_date` actually present in `fct_campaign_daily`.

    Read from the table rather than from configuration, because the two can
    disagree: `ADSIGNAL_START_DATE`/`ADSIGNAL_END_DATE` say which days a
    generator run was asked for, and this says which days survived the load.
    A model grounded on the first would confidently filter on days that were
    never written.
    """

    first_date: date
    last_date: date


# Matches `event_date` as a whole word, anywhere in a query: a WHERE clause, a
# SELECT list, a GROUP BY or an alias. It is not a parse, and it deliberately
# does not try to be one - `empty_result_notice` uses it only to decide
# whether one extra sentence of context is worth adding to an empty answer,
# and a false positive there costs a sentence.
_DATE_COLUMN_RE = re.compile(rf"\b{DATE_COLUMN}\b", re.IGNORECASE)


def describe_schema(coverage: DateCoverage | None) -> str:
    """The table shapes plus what period the data covers, for the prompt.

    The model previously saw `SCHEMA_DESCRIPTION` alone, which names tables
    and columns and says nothing about time. Asked "which channel spent the
    most last month?", it wrote a correct `CURRENT_DATE`-relative filter over
    a corpus that ends in the past, and the endpoint answered zero rows to a
    perfectly ordinary question.

    Carries table names, column names and two dates. No DSN, no key, no row.
    """
    return f"{SCHEMA_DESCRIPTION}\n{_coverage_note(coverage)}"


def _coverage_note(coverage: DateCoverage | None) -> str:
    """The paragraph describing the corpus's period, empty warehouse included.

    `None` gets a paragraph of its own rather than an omission: a prompt that
    silently drops the section reads, to the model, exactly like a warehouse
    whose range nobody mentioned, which is the state that caused the defect.
    """
    if coverage is None:
        return (
            f"Data coverage:\n"
            f"{COVERAGE_TABLE} currently holds no rows, so there is no date range to\n"
            f"anchor to and any filter on {DATE_COLUMN} matches nothing. Write the query\n"
            f"the question calls for regardless.\n"
        )
    first = coverage.first_date.isoformat()
    last = coverage.last_date.isoformat()
    return (
        f"Data coverage:\n"
        f"{COVERAGE_TABLE}.{DATE_COLUMN} runs from {first} to {last}, and every other\n"
        f"table describes those same days. This is a historical corpus: it does not\n"
        f"reach today's date, so a filter derived from CURRENT_DATE or NOW() matches\n"
        f"nothing at all.\n"
        f"Treat {last} as today. Anchor every relative expression in the question -\n"
        f'"last month", "recently", "this year", "the last 30 days", "yesterday" - to\n'
        f"DATE '{last}', and write that literal in place of CURRENT_DATE or NOW().\n"
    )


def empty_result_notice(sql: str, coverage: DateCoverage | None) -> str | None:
    """Why a query that filters on a date may legitimately have matched nothing.

    Returned only for a query whose text mentions `event_date`; for anything
    else there is nothing to say that "no rows" does not already say, and
    inventing a date explanation for a question about campaign names would be
    worse than silence.

    The caller decides that the result was empty. This decides whether an
    empty result deserves a sentence, and writes it.
    """
    if _DATE_COLUMN_RE.search(sql) is None:
        return None
    if coverage is None:
        return (
            f"{COVERAGE_TABLE} holds no rows yet, so nothing can match a filter on {DATE_COLUMN}."
        )
    return (
        f"{COVERAGE_TABLE} covers {coverage.first_date.isoformat()} to "
        f"{coverage.last_date.isoformat()}; an {DATE_COLUMN} filter outside "
        f"that range matches nothing."
    )
