"""The suite must never touch the read-only role a deployment is using.

A Postgres role is **cluster-wide**. The test databases are not: every
DSN-driven module gets its own, so nothing the suite writes can reach a
developer's data. The role had no such isolation, and
`ensure_readonly_role` resets the password every time it runs — so
`uv run pytest` silently invalidated whatever
`ADSIGNAL_POSTGRES_READONLY_DSN` the developer had configured, and `/ask`
answered `rule: "readonly_dsn_unverified"` to every question until the
bootstrap was run again. `tests/test_deploy_bootstrap.py` is worse: it runs
`DROP OWNED BY` against the role, which revokes the grants outright.

A reviewer recorded the hazard and nobody acted on it, so it is a test now.
Every check here is static — it reads the suite's own source and connects to
nothing.
"""

import ast
from pathlib import Path

import pytest

from adsignal.warehouse.ask_reads import READONLY_ROLE

_TESTS = Path(__file__).resolve().parent

# The helpers that create a role, reset its password or build a login DSN for
# it. Each takes the role name as its third positional parameter, defaulting
# to the deployment's; a call in this suite that leaves it out is the bug.
_ROLE_FUNCTIONS = frozenset({"ensure_readonly_role", "readonly_dsn_for"})
_ROLE_ARGUMENT_INDEX = 2

# What every module's own declaration must be called and must resolve to.
_CONSTANT_NAME = "TEST_READONLY_ROLE"


def _test_sources() -> list[Path]:
    """Every Python file in the suite, this one included."""
    return sorted(_TESTS.glob("*.py"))


def _declared_role(source: Path) -> str | None:
    """The literal default of this module's `TEST_READONLY_ROLE`, if it has one.

    Reads the `os.environ.get(..., "<name>")` call's second argument rather
    than the environment, so the assertion is about what the repository
    ships. An override set for one cluster is not a licence to ship a
    declaration that names the deployment's role.
    """
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if _CONSTANT_NAME not in names:
            continue
        call = node.value
        if isinstance(call, ast.Call) and len(call.args) == 2:
            default = call.args[1]
            if isinstance(default, ast.Constant) and isinstance(default.value, str):
                return default.value
    return None


def _role_calls(source: Path) -> list[tuple[str, ast.Call]]:
    """Every call in `source` to a helper that names a role."""
    tree = ast.parse(source.read_text())
    found: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in _ROLE_FUNCTIONS:
            assert name is not None
            found.append((name, node))
    return found


def test_the_suite_declares_a_role_of_its_own() -> None:
    """At least one module must declare it, or the other assertions test nothing."""
    declared = {p.name: _declared_role(p) for p in _test_sources()}
    named = {name: role for name, role in declared.items() if role is not None}
    assert named, f"no test module declares {_CONSTANT_NAME}"


@pytest.mark.parametrize("source", _test_sources(), ids=lambda p: p.name)
def test_no_declared_test_role_is_the_deployments_role(source: Path) -> None:
    role = _declared_role(source)
    if role is None:
        pytest.skip(f"{source.name} declares no {_CONSTANT_NAME}")
    assert role != READONLY_ROLE, (
        f"{source.name} points the suite at the deployment's role; "
        "running the tests would reset its password"
    )


def test_every_declaration_agrees() -> None:
    """Two spellings would mean two roles, and one of them unasserted."""
    declared = {p.name: _declared_role(p) for p in _test_sources()}
    roles = {role for role in declared.values() if role is not None}
    assert len(roles) == 1, f"the suite declares more than one test role: {sorted(roles)}"


@pytest.mark.parametrize("source", _test_sources(), ids=lambda p: p.name)
def test_every_role_helper_call_names_the_test_role(source: Path) -> None:
    """A call that omits the role argument gets the deployment's by default."""
    for name, call in _role_calls(source):
        given = None
        if len(call.args) > _ROLE_ARGUMENT_INDEX:
            given = call.args[_ROLE_ARGUMENT_INDEX]
        for keyword in call.keywords:
            if keyword.arg == "role":
                given = keyword.value
        assert given is not None, (
            f"{source.name}:{call.lineno} calls {name}() without a role, "
            f"so it uses the deployment's {READONLY_ROLE!r}"
        )
        assert isinstance(given, ast.Name) and given.id == _CONSTANT_NAME, (
            f"{source.name}:{call.lineno} passes a role that is not {_CONSTANT_NAME}"
        )


@pytest.mark.parametrize("source", _test_sources(), ids=lambda p: p.name)
def test_no_test_writes_the_deployments_role_name_as_a_literal(source: Path) -> None:
    """The role must arrive through the constant, never as a hardcoded string.

    Exact equality, not a substring: `adsignal_ro_test` and the table name
    `adsignal_ro_should_not_have_made_this` both contain the production name
    and neither is it.
    """
    tree = ast.parse(source.read_text())
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == READONLY_ROLE
    ]
    assert not offenders, f"{source.name} names {READONLY_ROLE!r} at lines {offenders}"
