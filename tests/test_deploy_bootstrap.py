"""A deployment that follows the documented steps can answer a question.

`tests/test_ask_reads.py` proves `ensure_readonly_role` grants what it says.
That test passed on every commit in which `/ask` refused every question in
production, because the test called the function itself — as did
`tests/conftest.py`'s `ask_client` fixture. Nothing in `src/` did, so a real
deployment created the role's DSN, passed the privilege probe, and then hit
`rule: "database_permission"` on every query.

The missing assertion is this one: start from a database in that exact state
— tables present, `adsignal_ro` present as a login role, no grants — run
only what `README.md`'s deployment section tells a deployer to run, and ask
a question. The suite is not allowed to do the granting on the deployment's
behalf anywhere in this module.
"""

import os
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql

from adsignal.api.app import create_app
from adsignal.cli.bootstrap_readonly_role import main
from adsignal.config import Settings, get_settings
from adsignal.narrative.sqlgen import SqlGenerator, SqlQuestion
from adsignal.warehouse.ask_reads import (
    READONLY_ROLE,
    forget_verified_dsns,
    readonly_dsn_for,
)
from adsignal.warehouse.loader import ensure_schema

BOOTSTRAP_DSN = os.environ.get(
    "ADSIGNAL_BOOTSTRAP_TEST_DSN",
    "postgresql://adsignal:adsignal@localhost:5432/adsignal_test_bootstrap",
)

# The role's password in this module. Not a credential: it reaches only a
# local test database, and it is written down here so that the "no secret is
# printed" assertion below has a literal string to look for.
RO_PASSWORD = "bootstrap-test-password-not-a-real-credential"

# The one question asked here. It reads a table the role is granted SELECT on
# only if the bootstrap ran, which is the whole point of the module.
GENERATED_SQL = "SELECT campaign_id, campaign_name FROM dim_campaign LIMIT 5"


class _FixedProvider:
    """A `SqlProvider` that always returns `GENERATED_SQL`."""

    def __init__(self) -> None:
        self.name = "stub"

    def generate_sql(self, request: SqlQuestion) -> str:
        return GENERATED_SQL


def _postgres_available() -> bool:
    try:
        with psycopg.connect(BOOTSTRAP_DSN, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _postgres_available(), reason="Postgres not reachable")


def _reset_to_pre_bootstrap_state() -> None:
    """Put the database in the state a deployment reaches without the bootstrap.

    Three things are true afterwards, and they are what `f65d139` produced on
    Render: the serving tables exist (the nightly load creates them), one row
    is loadable (the nightly load writes them), and `adsignal_ro` exists as a
    login role holding no grant on anything (`ADSIGNAL_POSTGRES_READONLY_DSN`
    was set by hand, and nothing ever ran `ensure_readonly_role`).

    `DROP OWNED BY` rather than `DROP ROLE`: it strips every privilege the
    role holds in this database while leaving the login itself, which is the
    only way to reach "the DSN connects but the grants were never applied".
    """
    ensure_schema(BOOTSTRAP_DSN)
    with psycopg.connect(BOOTSTRAP_DSN) as conn:
        conn.execute(
            "INSERT INTO dim_campaign VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            ("cmp_test_0001", "Spring Brand Lift", "adv_0001", "Northwind", "search"),
        )
        exists = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (READONLY_ROLE,)
        ).fetchone()
        verb = "CREATE ROLE"
        if exists is not None:
            # Strips every grant the role holds in this database and leaves
            # the login, which is the state this module needs.
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(READONLY_ROLE)))
            verb = "ALTER ROLE"
        # A password cannot be a bound parameter in role DDL, so it is
        # composed as a literal — the same thing `ensure_readonly_role` does.
        conn.execute(
            sql.SQL("{verb} {role} WITH LOGIN PASSWORD {password}").format(
                verb=sql.SQL(verb),
                role=sql.Identifier(READONLY_ROLE),
                password=sql.Literal(RO_PASSWORD),
            )
        )
        conn.commit()
    forget_verified_dsns()


def _ask(readonly_dsn: str) -> dict[str, Any]:
    """Ask one question of an app configured exactly as a deployment would be."""
    app = create_app(Settings(postgres_dsn=BOOTSTRAP_DSN, postgres_readonly_dsn=readonly_dsn))
    with TestClient(app) as client:
        client.app.state.sql_generator = SqlGenerator(_FixedProvider())
        response = client.post("/ask", json={"question": "name a few campaigns"})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


@pytest.fixture
def deployment_environment(monkeypatch: pytest.MonkeyPatch) -> str:
    """The environment a deployer sets, and the read-only DSN it contains.

    `get_settings` is process-cached, so the cache is cleared on the way in
    and on the way out; otherwise this module's DSNs would leak into every
    later test that reads settings from the environment.
    """
    readonly_dsn = readonly_dsn_for(BOOTSTRAP_DSN, RO_PASSWORD)
    monkeypatch.setenv("ADSIGNAL_POSTGRES_DSN", BOOTSTRAP_DSN)
    monkeypatch.setenv("ADSIGNAL_POSTGRES_READONLY_DSN", readonly_dsn)
    get_settings.cache_clear()
    yield readonly_dsn
    get_settings.cache_clear()


def test_a_question_is_refused_until_the_bootstrap_runs_and_answered_after(
    deployment_environment: str,
) -> None:
    """The four-instance bug, end to end, in one test.

    The first half is the bug as the project owner met it in a browser: a
    configured deployment refusing every question with `database_permission`.
    The second half runs the documented command and nothing else, and the
    same question is answered.
    """
    readonly_dsn = deployment_environment
    _reset_to_pre_bootstrap_state()

    refused = _ask(readonly_dsn)
    assert refused["rule"] == "database_permission", refused
    assert refused["rows"] == []

    assert main() == 0

    forget_verified_dsns()
    answered = _ask(readonly_dsn)
    assert answered["refusal"] is None, answered
    assert answered["rule"] is None
    assert answered["columns"] == ["campaign_id", "campaign_name"]
    assert answered["rows"], "the bootstrap ran but the query returned nothing"


def test_running_the_bootstrap_twice_is_fine(deployment_environment: str) -> None:
    """A deploy hook runs on every deploy, not only the first."""
    _reset_to_pre_bootstrap_state()
    assert main() == 0
    assert main() == 0


def test_the_bootstrap_prints_no_password(
    deployment_environment: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Global constraint: no secret reaches a log line, a row or a response."""
    _reset_to_pre_bootstrap_state()
    assert main() == 0
    captured = capsys.readouterr()
    assert RO_PASSWORD not in captured.out
    assert RO_PASSWORD not in captured.err


def test_an_unset_readonly_dsn_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence here would be the same defect one layer up: nothing ran, exit 0."""
    monkeypatch.setenv("ADSIGNAL_POSTGRES_DSN", BOOTSTRAP_DSN)
    monkeypatch.setenv("ADSIGNAL_POSTGRES_READONLY_DSN", "")
    get_settings.cache_clear()
    try:
        assert main() == 2
    finally:
        get_settings.cache_clear()
