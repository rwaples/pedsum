"""Direct unit tests for the validate stderr summary formatter.

Covers the four rendering branches of ``_format_check_summary`` (PASS /
PASS-with-note / FAIL / SKIP-with-note), the finding-total
accounting (only FAIL contributes), and the ``_summarize_findings``
>5-finding truncation.
"""

from __future__ import annotations

from pathlib import Path

from pedsum.checks import CheckResult, CheckStatus, Finding, _summarize_findings
from pedsum.report import _format_check_summary


def test_format_check_summary_renders_all_status_combinations():
    """Each status / note combination renders its own line shape."""
    results = [
        CheckResult(name="required_columns", status=CheckStatus.PASS),
        CheckResult(
            name="sex_role_consistency",
            status=CheckStatus.PASS,
            note="1 overridden from role",
        ),
        CheckResult(name="duplicate_ids", status=CheckStatus.FAIL, findings=[Finding()] * 3),
        CheckResult(
            name="negative_ids",
            status=CheckStatus.SKIP,
            note="id_dtype failed",
        ),
    ]
    out = _format_check_summary(Path("p.tsv"), n_total=5, results=results)
    # PASS line carries no parenthetical.
    assert "required columns present" in out
    assert " PASS\n" in out  # plain PASS row exists somewhere
    # Annotated PASS (0.9 sex_role_consistency override): trailing note.
    assert "PASS (1 overridden from role)" in out
    # FAIL line carries the count.
    assert "no duplicate IDs " in out
    assert "FAIL (3)" in out
    # SKIP line carries the note.
    assert "SKIP (id_dtype failed)" in out


def test_format_check_summary_total_findings_only_counts_fail():
    """``result: N finding(s)`` only counts FAIL rows, not an annotated PASS."""
    results = [
        CheckResult(name="duplicate_ids", status=CheckStatus.FAIL, findings=[Finding()] * 2),
        CheckResult(
            name="sex_role_consistency",
            status=CheckStatus.PASS,
            note="1 overridden from role",
        ),
        CheckResult(
            name="acyclic",
            status=CheckStatus.SKIP,
            note="parent dtype failed",
        ),
    ]
    out = _format_check_summary(Path("p.tsv"), n_total=5, results=results)
    assert "result: 2 finding(s)" in out


def test_summarize_findings_truncates_above_five():
    """8 findings → sample shows first 5 plus ``(and 3 more)`` suffix."""
    findings = [Finding(check="negative_ids", id=i, row=i - 1, detail=f"id={i}") for i in range(1, 9)]
    out = _summarize_findings(findings, str)
    assert out.startswith("negative_ids: 8 finding(s) — ")
    # First five rows present.
    for i in range(1, 6):
        assert f"row {i - 1} (id={i})" in out
    # Truncation suffix.
    assert "(and 3 more)" in out
    # Findings past the fifth are NOT spelled out individually.
    assert "row 5 (id=6)" not in out
