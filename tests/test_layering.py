"""The project's dependency arrows, enforced rather than documented.

`ml/gate.py` imported `adsignal.config.get_settings` and
`adsignal.warehouse.reader.read_campaign_daily`, which made the whole ML
layer impossible to test or reuse without a database and an environment. A
prose rule did not catch that, so this does: every module under each package
is parsed and its imports checked against the packages it is allowed to
reach.
"""

import ast
from pathlib import Path

import pytest

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "adsignal"

# package -> the sibling packages it must never import.
FORBIDDEN_IMPORTS = {
    "ml": ("config", "warehouse", "pipeline", "api"),
    "narrative": ("warehouse", "api"),
    "generator": ("config", "warehouse", "pipeline", "api", "ml"),
}


def _all_imports(source: Path) -> set[str]:
    """Every module name this file imports, by either import form."""
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _adsignal_imports(source: Path) -> set[str]:
    """Only the project's own modules, which are the ones layering constrains."""
    return {name for name in _all_imports(source) if name.startswith("adsignal.")}


def _modules_under(package: str) -> list[Path]:
    return sorted((_SOURCE_ROOT / package).rglob("*.py"))


@pytest.mark.parametrize("package", sorted(FORBIDDEN_IMPORTS))
def test_a_package_never_imports_the_layers_below_it(package: str) -> None:
    forbidden = FORBIDDEN_IMPORTS[package]
    violations = [
        f"{source.relative_to(_SOURCE_ROOT)} imports {name}"
        for source in _modules_under(package)
        for name in _adsignal_imports(source)
        if name.split(".")[1] in forbidden
    ]
    assert violations == []


def test_the_api_layer_issues_no_sql() -> None:
    """`warehouse/` owns all database access; `api/` reads through it."""
    offenders = [
        str(source.relative_to(_SOURCE_ROOT))
        for source in _modules_under("api")
        if any(name.startswith("psycopg") for name in _all_imports(source))
    ]
    assert offenders == []
