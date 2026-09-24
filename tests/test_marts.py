from pathlib import Path

from adsignal.warehouse.duckdb_io import open_duckdb, register_sources, run_dbt

DBT_DIR = Path(__file__).resolve().parents[1] / "dbt"


def _build(tmp_path, small_dataset):
    catalog, events_path, labels_path = small_dataset
    duckdb_path = tmp_path / "marts.duckdb"
    with open_duckdb(duckdb_path) as conn:
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)
    run_dbt(project_dir=DBT_DIR, duckdb_path=duckdb_path, command="build")
    return duckdb_path


def test_mart_aggregates_creatives_to_campaign_days(tmp_path, small_dataset) -> None:
    duckdb_path = _build(tmp_path, small_dataset)
    with open_duckdb(duckdb_path) as conn:
        duplicates = conn.execute(
            """
            SELECT count(*) FROM (
                SELECT event_date, campaign_id
                FROM main_marts.fct_campaign_daily
                GROUP BY 1, 2 HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
        assert duplicates == 0


def test_mart_totals_match_the_raw_events(tmp_path, small_dataset) -> None:
    duckdb_path = _build(tmp_path, small_dataset)
    with open_duckdb(duckdb_path) as conn:
        raw = conn.execute("SELECT sum(impressions), sum(spend) FROM raw_events").fetchone()
        mart = conn.execute(
            "SELECT sum(impressions), sum(spend) FROM main_marts.fct_campaign_daily"
        ).fetchone()
        assert raw[0] == mart[0]
        assert abs(raw[1] - mart[1]) < 0.01


def test_zero_denominator_metrics_are_null(tmp_path, small_dataset) -> None:
    """Review Focus 3: CPA with zero conversions, CTR with zero impressions,
    and ROAS with zero spend must be NULL, never 0 and never an error."""
    duckdb_path = _build(tmp_path, small_dataset)
    with open_duckdb(duckdb_path) as conn:
        bad_cpa = conn.execute(
            "SELECT count(*) FROM main_marts.fct_campaign_daily "
            "WHERE conversions = 0 AND cpa IS NOT NULL"
        ).fetchone()[0]
        bad_ctr = conn.execute(
            "SELECT count(*) FROM main_marts.fct_campaign_daily "
            "WHERE impressions = 0 AND ctr IS NOT NULL"
        ).fetchone()[0]
        bad_roas = conn.execute(
            "SELECT count(*) FROM main_marts.fct_campaign_daily "
            "WHERE spend = 0 AND roas IS NOT NULL"
        ).fetchone()[0]
        assert (bad_cpa, bad_ctr, bad_roas) == (0, 0, 0)


def test_anomaly_labels_join_onto_the_mart(tmp_path, small_dataset) -> None:
    duckdb_path = _build(tmp_path, small_dataset)
    with open_duckdb(duckdb_path) as conn:
        flagged = conn.execute(
            "SELECT count(*) FROM main_marts.fct_campaign_daily WHERE is_anomaly"
        ).fetchone()[0]
        labels = conn.execute(
            "SELECT count(DISTINCT (campaign_id, event_date)) FROM raw_anomaly_labels"
        ).fetchone()[0]
        assert flagged == labels
        orphans = conn.execute(
            "SELECT count(*) FROM main_marts.fct_campaign_daily "
            "WHERE is_anomaly AND anomaly_cause IS NULL"
        ).fetchone()[0]
        assert orphans == 0
