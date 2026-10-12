"""A name one pedsum module imports from another is public (no underscore); tests may still import privates."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "pedsum"


def private_cross_module_imports() -> list[str]:
    """Each ``from pedsum.x import _y`` inside the package, as ``path:line: name``."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "pedsum":
                found += [
                    f"{path.relative_to(PACKAGE.parent)}:{node.lineno}: {node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("_") and not alias.name.startswith("__")
                ]
    return found


def test_no_module_imports_a_private_name_from_another():
    """Within pedsum/, cross-module imports name only public symbols."""
    assert private_cross_module_imports() == []
