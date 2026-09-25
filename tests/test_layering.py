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
    """Every module name this file imports, by either import form.

    Relative imports are resolved to their absolute name first. `from
    ..config import get_settings` names the same module as `from
    adsignal.config import ...` and is exactly the violation this test
    exists to catch, so it must not slip through as the bare name "config".
    """
    tree = ast.parse(source.read_text())
    package = _package_of(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(_resolve(node, package))
    return imported


def _package_of(source: Path) -> list[str]:
    """The dotted package parts containing `source`, rooted at `adsignal`."""
    relative = source.relative_to(_SOURCE_ROOT.parent)
    parts = list(relative.parts[:-1])
    if relative.stem != "__init__":
        return parts
    return parts


def _resolve(node: ast.ImportFrom, package: list[str]) -> str:
    """The absolute module name `node` refers to, resolving any relative form."""
    if not node.level:
        return node.module or ""
    base = package[: len(package) - node.level + 1]
    return ".".join([*base, node.module] if node.module else base)


def _adsignal_imports(source: Path) -> set[str]:
    """Only the project's own modules, which are the ones layering constrains."""
    return {name for name in _all_imports(source) if name.startswith("adsignal.")}


def _layer_of(module: str) -> str:
    """The layer a dotted `adsignal.<layer>...` name belongs to, or "" for none.

    Returns "" rather than raising on a bare `import adsignal`, which has no
    layer to constrain.
    """
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def _modules_under(package: str) -> list[Path]:
    return sorted((_SOURCE_ROOT / package).rglob("*.py"))


@pytest.mark.parametrize("package", sorted(FORBIDDEN_IMPORTS))
def test_a_package_never_imports_the_layers_below_it(package: str) -> None:
    forbidden = FORBIDDEN_IMPORTS[package]
    violations = [
        f"{source.relative_to(_SOURCE_ROOT)} imports {name}"
        for source in _modules_under(package)
        for name in _adsignal_imports(source)
        if _layer_of(name) in forbidden
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
