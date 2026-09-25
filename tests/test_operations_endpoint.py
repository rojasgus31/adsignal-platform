"""`GET /operations`."""


def test_operations_returns_recent_runs(client) -> None:
    response = client.get("/operations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_operations_is_bounded(client) -> None:
    """An unbounded limit on a public endpoint is a denial-of-service lever."""
    assert client.get("/operations?limit=100000").status_code == 422
    assert client.get("/operations?limit=0").status_code == 422


def test_operations_on_a_fresh_database_is_200_and_empty(empty_client) -> None:
    response = empty_client.get("/operations")
    assert response.status_code == 200
    assert response.json() == []


def test_operations_with_no_tables_at_all_is_200(dropped_tables_client) -> None:
    """The contract every read endpoint in this API holds."""
    assert dropped_tables_client.get("/operations").status_code == 200


def test_no_row_exposes_a_connection_string(client) -> None:
    """detail carries exception type names only."""
    body = client.get("/operations").text
    assert "postgresql://" not in body
    assert "password" not in body.lower()
