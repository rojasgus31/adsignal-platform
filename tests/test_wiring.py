"""Functions that are only reachable from the test suite.

Four times now, this project has built a function, tested it, documented it,
and never called it from `src/`:

- `loader.load_dim_campaign`, called by no Dagster asset: the dashboard
  showed `cmp_0058_006` where a campaign name belonged.
- `campaign_reads.list_campaigns`, called by no route: the names died at the
  API boundary.
- `narrative_store.read_latest_narrative`, not called by `/benchmark`: the
  Overview composed its own paragraph instead of serving the stored one.
- `ask_reads.ensure_readonly_role`, called by nothing at all: `/ask` refused
  every question with `rule: "database_permission"`.

Every one of them passed its own unit test, because the test called it
directly — and in the fourth case `tests/conftest.py` called it during
fixture setup, which is production's job done by the suite. A fixture that
does what a deployment is supposed to do makes the test pass and the product
broken.

This module is the crude, explicit guard: a hand-maintained list of
functions whose whole purpose is to be called by shipped code, checked
against every call expression under `src/adsignal`. It is deliberately a
list rather than an inference — "is this function meant to be wired in?" is
a judgement, and writing the judgement down is the point.

Adding a genuinely internal helper here would be wrong. Removing an entry
because it started failing would be the exact mistake this file exists to
catch.
"""

import ast
from pathlib import Path

import pytest

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "adsignal"

# module path (relative to `src/adsignal`) -> functions in it that shipped
# code must call. The module's own file does not count as a caller.
MUST_BE_CALLED_FROM_SRC = {
    "warehouse/ask_reads.py": ("ensure_readonly_role",),
    "warehouse/loader.py": ("load_dim_campaign",),
    "warehouse/campaign_reads.py": ("list_campaigns",),
    "warehouse/narrative_store.py": ("read_latest_narrative",),
}


def _called_names(source: Path) -> set[str]:
    """Every name this file calls, by either `name(...)` or `obj.name(...)`.

    Only call expressions count. Importing a function and never calling it
    is exactly the state this module exists to detect, so an import is not
    evidence of a caller.
    """
    tree = ast.parse(source.read_text())
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)
    return called


def _callers_outside(defining_module: Path) -> set[str]:
    """Every function name called anywhere under `src/adsignal` but that file."""
    called: set[str] = set()
    for source in sorted(_SOURCE_ROOT.rglob("*.py")):
        if source == defining_module:
            continue
        called |= _called_names(source)
    return called


@pytest.mark.parametrize("module_path", sorted(MUST_BE_CALLED_FROM_SRC))
def test_a_function_meant_for_production_has_a_caller_in_src(module_path: str) -> None:
    defining_module = _SOURCE_ROOT / module_path
    callers = _callers_outside(defining_module)
    unwired = [name for name in MUST_BE_CALLED_FROM_SRC[module_path] if name not in callers]
    assert unwired == [], (
        f"{module_path}: {unwired} is called only by tests. "
        "Shipped code never runs it, so every test covering it proves nothing "
        "about a real deployment."
    )
