from adsignal.warehouse.duckdb_io import open_duckdb, register_sources


def test_register_sources_exposes_every_table(tmp_path, small_dataset) -> None:
    catalog, events_path, labels_path = small_dataset
    with open_duckdb(tmp_path / "test.duckdb") as conn:
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        assert {
            "raw_events",
            "raw_anomaly_labels",
            "raw_campaigns",
            "raw_creatives",
            "raw_advertisers",
        } <= tables


def test_registered_row_counts_match_the_catalog(tmp_path, small_dataset) -> None:
    catalog, events_path, labels_path = small_dataset
    with open_duckdb(tmp_path / "test.duckdb") as conn:
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)
        campaigns = conn.execute("SELECT count(*) FROM raw_campaigns").fetchone()[0]
        creatives = conn.execute("SELECT count(*) FROM raw_creatives").fetchone()[0]
        assert campaigns == len(catalog.campaigns)
        assert creatives == len(catalog.creatives)


def test_register_sources_is_idempotent(tmp_path, small_dataset) -> None:
    catalog, events_path, labels_path = small_dataset
    with open_duckdb(tmp_path / "test.duckdb") as conn:
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)
        first = conn.execute("SELECT count(*) FROM raw_events").fetchone()[0]
        register_sources(conn, events_path=events_path, labels_path=labels_path, catalog=catalog)
        second = conn.execute("SELECT count(*) FROM raw_events").fetchone()[0]
        assert first == second
