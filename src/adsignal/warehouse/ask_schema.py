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

Pure data. This module imports nothing and opens nothing, so it is safe for
the API layer to import and pass to a model provider.
"""

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
