def test_health_reports_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_endpoints_return_empty_not_error_on_a_fresh_database(empty_client) -> None:
    """Review Focus 4: a visitor arriving before the first pipeline run.

    Every read endpoint must answer 200 with an empty payload, never 500.
    """
    for path in ("/campaigns", "/anomalies", "/models"):
        response = empty_client.get(path)
        assert response.status_code == 200, f"{path} returned {response.status_code}"
        assert response.json() == [] or response.json() == {"items": []}

    benchmark = empty_client.get("/benchmark")
    assert benchmark.status_code == 200
    assert benchmark.json()["runs"] == []


def test_endpoints_return_empty_not_error_with_no_tables_at_all(dropped_tables_client) -> None:
    """Fix-round regression test for the Critical finding on the first review.

    `empty_client` only ever exercises empty-but-existing tables, because its
    setup runs `ensure_*_schema` before truncating. A database that has never
    had the pipeline run against it - a serving database just recreated for
    pgvector, say - has no tables at all, which is a step further and is what
    actually 500'd in review: `GET /campaigns` raised
    `psycopg.errors.UndefinedTable: relation "fct_campaign_daily" does not
    exist`. `dropped_tables_client` reproduces that by dropping every table
    rather than truncating it.
    """
    for path in ("/health", "/campaigns", "/anomalies", "/models"):
        response = dropped_tables_client.get(path)
        assert response.status_code == 200, f"{path} returned {response.status_code}"

    assert dropped_tables_client.get("/campaigns").json() == []
    assert dropped_tables_client.get("/anomalies").json() == []
    assert dropped_tables_client.get("/models").json() == []

    benchmark = dropped_tables_client.get("/benchmark")
    assert benchmark.status_code == 200
    assert benchmark.json()["runs"] == []

    search = dropped_tables_client.post("/search", json={"query": "anything", "limit": 3})
    assert search.status_code == 200
    assert search.json()["hits"] == []

    # An unknown campaign/campaign-day on a table-less database is still a
    # 404, not a 500 - the same "not found" answer as on a populated database.
    unknown_metrics = dropped_tables_client.get("/campaigns/does-not-exist/metrics")
    assert unknown_metrics.status_code == 404

    unknown_anomaly = dropped_tables_client.get("/anomalies/does-not-exist/2024-01-01")
    assert unknown_anomaly.status_code == 404


def test_campaigns_lists_known_campaigns(client) -> None:
    body = client.get("/campaigns").json()
    assert body
    assert {"campaign_id", "channel"} <= set(body[0])


def test_campaign_metrics_returns_a_time_series(client, seeded_campaign_id) -> None:
    body = client.get(f"/campaigns/{seeded_campaign_id}/metrics").json()
    assert body["campaign_id"] == seeded_campaign_id
    assert len(body["points"]) > 0
    assert {"event_date", "spend", "conversions"} <= set(body["points"][0])


def test_unknown_campaign_is_a_404_not_a_500(client) -> None:
    assert client.get("/campaigns/does-not-exist/metrics").status_code == 404


def test_anomalies_returns_predictions(client) -> None:
    body = client.get("/anomalies").json()
    assert isinstance(body, list)
    if body:
        assert {"campaign_id", "event_date", "cause", "severity", "escalate"} <= set(body[0])


def test_models_returns_metric_history_newest_first(client) -> None:
    body = client.get("/models").json()
    assert isinstance(body, list)
    if len(body) > 1:
        assert body[0]["trained_at"] >= body[1]["trained_at"]


def test_search_returns_ranked_hits(client) -> None:
    response = client.post("/search", json={"query": "winter clearance", "limit": 3})
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert len(hits) <= 3
    if len(hits) > 1:
        assert hits[0]["distance"] <= hits[1]["distance"]


def test_search_rejects_a_blank_query(client) -> None:
    assert client.post("/search", json={"query": "  "}).status_code == 422


def test_search_limit_is_bounded(client) -> None:
    """An unbounded limit is a denial-of-service lever on a public demo."""
    assert client.post("/search", json={"query": "sale", "limit": 10_000}).status_code == 422


def test_openapi_schema_is_served(client) -> None:
    assert client.get("/openapi.json").status_code == 200


def test_no_endpoint_response_contains_a_secret(client) -> None:
    """Keys must never reach a response body."""
    for path in ("/health", "/models", "/benchmark"):
        assert "api_key" not in client.get(path).text.lower()
