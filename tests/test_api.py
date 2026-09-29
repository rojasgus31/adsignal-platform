from fastapi.testclient import TestClient

from adsignal.warehouse.paging import (
    DEFAULT_RUN_PAGE_LIMIT,
    DEFAULT_SERVING_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
)


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
    # No narrative has been generated yet, so the field is explicitly null.
    # Never an empty string, which a dashboard would render as a blank
    # paragraph rather than as "nothing to show".
    assert benchmark.json()["latest_narrative"] is None


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
    assert benchmark.json()["latest_narrative"] is None

    search = dropped_tables_client.post("/search", json={"query": "anything", "limit": 3})
    assert search.status_code == 200
    assert search.json()["hits"] == []

    # An unknown campaign/campaign-day on a table-less database is still a
    # 404, not a 500 - the same "not found" answer as on a populated database.
    unknown_metrics = dropped_tables_client.get("/campaigns/does-not-exist/metrics")
    assert unknown_metrics.status_code == 404

    unknown_anomaly = dropped_tables_client.get("/anomalies/does-not-exist/2024-01-01")
    assert unknown_anomaly.status_code == 404

    # `/ask` is the first thing a visitor hits on a freshly deployed demo, so
    # it must answer rather than 500. On this deployment no model key and no
    # read-only DSN are configured, which it reports as a plain 200 with an
    # explanation rather than an error or an invented answer.
    ask = dropped_tables_client.post("/ask", json={"question": "how many anomalies?"})
    assert ask.status_code == 200, ask.text
    body = ask.json()
    assert body["rule"] == "not_configured"
    assert body["rows"] == []
    assert body["sql"] is None


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


def test_benchmark_serves_the_stored_narrative(client, seeded_narrative) -> None:
    """The prose the dashboard shows is the row in `narrative_runs`, verbatim.

    Asserting equality against the seeded text - not just "some non-empty
    string" - is what keeps the API from regenerating or reformatting the
    narrative: whatever provider wrote the row is what a reader sees.
    """
    body = client.get("/benchmark").json()
    narrative = body["latest_narrative"]
    assert narrative is not None
    assert narrative["text"] == seeded_narrative.text
    assert narrative["provider"] == seeded_narrative.provider
    assert narrative["as_of"] == "2024-06-02"
    assert narrative["generated_at"]


def test_benchmark_narrative_carries_no_credential(client) -> None:
    """`provider` is a name only - no key, no endpoint URL that could carry one."""
    narrative = client.get("/benchmark").json()["latest_narrative"]
    assert narrative["provider"].isidentifier()
    assert "://" not in narrative["provider"]


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


def test_ask_is_rate_limited_per_ip(client) -> None:
    """One of the two live paths must be bounded; `/search` is the other.

    The limiter is checked before anything else, so it bounds the endpoint
    whether or not a model is configured. Everything else about `/ask` —
    generation, validation, execution and the refusals — lives in
    `test_ask_endpoint.py`.
    """
    statuses = [
        client.post("/ask", json={"question": f"how many anomalies? {index}"}).status_code
        for index in range(30)
    ]
    assert 429 in statuses, "no request was ever rate-limited"


def test_search_is_rate_limited_per_ip(client) -> None:
    """Finding 6: `/search` runs ONNX inference on caller text per request.

    It was the one endpoint left unguarded while `/ask`, which only runs a
    bounded indexed query, was throttled - so the cheapest denial-of-service
    lever against the demo was the endpoint that does the most work.
    """
    statuses = [
        client.post("/search", json={"query": "winter coat", "limit": 3}).status_code
        for _ in range(30)
    ]
    assert 429 in statuses, "no request was ever rate-limited"
    assert statuses[0] == 200, "the first request should have been served"


def test_search_and_ask_do_not_share_a_rate_limit_budget(client) -> None:
    """Separate buckets: draining one endpoint must not lock out the other."""
    for _ in range(30):
        client.post("/search", json={"query": "winter coat", "limit": 3})

    answered = client.post("/ask", json={"question": "how many anomalies?"})
    assert answered.status_code == 200, answered.text


def test_models_is_bounded_and_paginates(client) -> None:
    """Finding 10: `/anomalies` and `/models` returned every row ever written,
    while `/search` and `/ask` bounded theirs precisely because an unbounded
    limit is a denial-of-service lever."""
    page = client.get("/models", params={"limit": 1})
    assert page.status_code == 200
    assert len(page.json()) == 1

    second = client.get("/models", params={"limit": 1, "offset": 1})
    assert second.status_code == 200
    assert second.json() != page.json()


def test_anomalies_is_bounded(client) -> None:
    page = client.get("/anomalies", params={"limit": 1})
    assert page.status_code == 200
    assert len(page.json()) <= 1


def test_the_collection_bounds_are_enforced(client) -> None:
    """The contract is a 422, not a silently honoured unbounded request."""
    for path in ("/models", "/anomalies"):
        assert client.get(path, params={"limit": 0}).status_code == 422
        assert client.get(path, params={"limit": MAX_PAGE_LIMIT + 1}).status_code == 422
        assert client.get(path, params={"offset": -1}).status_code == 422


def test_the_collections_default_to_a_bounded_page(client) -> None:
    """A caller that asks for nothing still gets a bounded response."""
    for path in ("/models", "/anomalies"):
        response = client.get(path)
        assert response.status_code == 200
        assert len(response.json()) <= DEFAULT_SERVING_PAGE_LIMIT


def test_two_forwarded_callers_do_not_share_a_rate_limit_budget(monkeypatch) -> None:
    """Review Focus 5, end to end. Behind a proxy each visitor needs its own budget."""
    monkeypatch.setenv("ADSIGNAL_TRUSTED_PROXY_HOPS", "1")
    # This test builds its app from the environment, to exercise the setting
    # it just set. A `.env` carrying a real `ADSIGNAL_GEMINI_API_KEY` would
    # otherwise make every `/ask` below a live, billed model call; blanking
    # it keeps the test about the rate limiter, which is all it asserts.
    monkeypatch.setenv("ADSIGNAL_GEMINI_API_KEY", "")
    from adsignal.api.app import create_app

    with TestClient(create_app()) as proxied:
        first = [
            proxied.post(
                "/ask",
                json={"question": "how many anomalies?"},
                headers={"X-Forwarded-For": "198.51.100.1"},
            ).status_code
            for _ in range(30)
        ]
        assert 429 in first, "the first caller was never limited"

        second = proxied.post(
            "/ask",
            json={"question": "how many anomalies?"},
            headers={"X-Forwarded-For": "198.51.100.2"},
        )
        assert second.status_code == 200, "a second caller inherited the first's budget"


def test_campaigns_carry_human_names(client) -> None:
    """The spec's success criterion: every table shows names, not identifiers.

    The seed loads `dim_campaign` from the same DuckDB file as the mart, so
    every row here has a dimension row and the names are the generated ones
    (`build_catalog` never emits a name starting with `cmp_`/`adv_`), not the
    identifier fallback `test_a_campaign_missing_from_the_dimension_*`
    covers.
    """
    body = client.get("/campaigns").json()
    assert body
    assert {"campaign_id", "campaign_name", "advertiser_id", "advertiser_name"} <= set(body[0])
    for row in body:
        assert not row["campaign_name"].startswith("cmp_")
        assert not row["advertiser_name"].startswith("adv_")


def test_anomalies_carry_a_campaign_name(client) -> None:
    """Task 10 groups the anomalies table by campaign name, so the name has to
    arrive on the anomaly row itself rather than in a second request."""
    body = client.get("/anomalies", params={"limit": 5}).json()
    assert body
    for row in body:
        assert row["campaign_name"]
        assert not row["campaign_name"].startswith("cmp_")


def test_one_anomaly_carries_a_campaign_name(client) -> None:
    page = client.get("/anomalies", params={"limit": 1}).json()
    assert page
    first = page[0]
    single = client.get(f"/anomalies/{first['campaign_id']}/{first['event_date']}")
    assert single.status_code == 200
    assert single.json()["campaign_name"] == first["campaign_name"]


def test_a_campaign_missing_from_the_dimension_falls_back_to_its_id(
    client, fact_row_without_dimension
) -> None:
    """A dashboard showing an id is worse than nothing; a dashboard silently
    dropping the row is worse still. The id is what the name fields carry when
    the dimension has no row for the campaign."""
    campaign_id, advertiser_id, _ = fact_row_without_dimension

    rows = client.get("/campaigns").json()
    orphan = next((row for row in rows if row["campaign_id"] == campaign_id), None)
    assert orphan is not None, "the campaign vanished from /campaigns"
    assert orphan["campaign_name"] == campaign_id
    assert orphan["advertiser_name"] == advertiser_id


def test_an_anomaly_missing_from_the_dimension_falls_back_to_its_id(
    client, fact_row_without_dimension
) -> None:
    campaign_id, _, event_date = fact_row_without_dimension

    page = client.get("/anomalies", params={"limit": 1}).json()
    assert page, "the orphan anomaly vanished from /anomalies"
    assert page[0]["campaign_id"] == campaign_id
    assert page[0]["campaign_name"] == campaign_id

    single = client.get(f"/anomalies/{campaign_id}/{event_date}")
    assert single.status_code == 200
    assert single.json()["campaign_name"] == campaign_id


def test_the_name_fields_survive_a_fresh_database(empty_client) -> None:
    """Fresh-database contract for the two name-carrying payloads: 200 with an
    empty list, never a 500 and never a row missing its new keys."""
    for path in ("/campaigns", "/anomalies"):
        response = empty_client.get(path)
        assert response.status_code == 200, f"{path} returned {response.status_code}"
        assert response.json() == []


def test_the_name_fields_survive_a_database_with_no_tables(dropped_tables_client) -> None:
    """`dropped_tables_client` drops `dim_campaign` too, so this covers the
    join's own table being absent, not just empty."""
    for path in ("/campaigns", "/anomalies"):
        response = dropped_tables_client.get(path)
        assert response.status_code == 200, f"{path} returned {response.status_code}"
        assert response.json() == []


def test_each_collection_declares_the_page_default_its_own_reader_uses(client) -> None:
    """The two `DEFAULT_*_PAGE_LIMIT` constants differ on purpose, and each
    endpoint has to advertise the one belonging to the reader behind it: the
    operations log pages at 50, the serving reads at 100.

    The published defaults are asserted against the literals 50 and 100, not
    only against the constants. Comparing an endpoint's declared default to
    the constant it is populated from lets both sides move together - editing
    `paging.py` from 50 to 75 would change what `/operations` promises every
    client while the test stayed green (Task 9b review, Finding 1). The
    constants are then checked against the same literals, so a rename that
    repointed an endpoint at the other reader's default still fails.
    """
    paths = client.get("/openapi.json").json()["paths"]

    def declared_default(path: str) -> int:
        parameters = paths[path]["get"]["parameters"]
        limit = next(parameter for parameter in parameters if parameter["name"] == "limit")
        return int(limit["schema"]["default"])

    assert declared_default("/operations") == 50
    assert declared_default("/anomalies") == 100
    assert declared_default("/models") == 100

    assert DEFAULT_RUN_PAGE_LIMIT == 50
    assert DEFAULT_SERVING_PAGE_LIMIT == 100
    assert MAX_PAGE_LIMIT == 500
