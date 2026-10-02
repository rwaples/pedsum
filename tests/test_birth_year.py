"""``summarize --birth-year-col`` naming a missing column fails validation.

The Ne_H tests that used this flag moved to ``test_effective_size_cmd.py``
with the estimators (ADR 0004).
"""

from __future__ import annotations

from conftest import EXAMPLE
from conftest import run_pedsum as _run


def test_birth_year_missing_column_errors(tmp_path):
    """Naming a non-existent column fails with the standard missing-cols error."""
    base = tmp_path / "p"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(base),
            "--birth-year-col",
            "does_not_exist",
        ]
    )
    assert res.returncode == 1
    assert "missing required columns" in res.stderr
    assert "does_not_exist" in res.stderr
