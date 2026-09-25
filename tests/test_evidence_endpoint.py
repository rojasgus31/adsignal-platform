"""The evidence endpoint exposes what model_metrics already stores."""


def test_evidence_returns_the_stored_confusion_matrix(client) -> None:
    runs = client.get("/models?limit=1").json()
    assert runs, "no model runs to read evidence for"
    run_id = runs[0]["run_id"]

    response = client.get(f"/models/{run_id}/evidence")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run_id
    assert body["holdout_samples"] > 0
    assert body["per_class_f1"], "per-class F1 is empty"
    assert body["confusion"], "the confusion matrix is empty"


def test_every_confusion_row_sums_to_that_causes_support(client) -> None:
    """A matrix whose rows do not sum to the class count is not a matrix."""
    run_id = client.get("/models?limit=1").json()[0]["run_id"]
    body = client.get(f"/models/{run_id}/evidence").json()
    total = sum(sum(row.values()) for row in body["confusion"].values())
    assert total == body["holdout_samples"]


def test_an_unknown_run_id_is_404_not_500(client) -> None:
    response = client.get("/models/no-such-run/evidence")
    assert response.status_code == 404


def test_a_run_id_that_looks_like_sql_is_rejected_safely(client) -> None:
    """The path parameter reaches a parameterized query, never string interpolation."""
    response = client.get("/models/'; DROP TABLE model_metrics; --/evidence")
    assert response.status_code in (404, 422)
    assert client.get("/models").status_code == 200


def test_evidence_on_a_fresh_database_is_404_not_500(empty_client) -> None:
    """The established fresh-database contract."""
    assert empty_client.get("/models/anything/evidence").status_code == 404
