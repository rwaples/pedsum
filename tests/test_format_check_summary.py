"""Direct unit tests for the validate stderr summary formatter.

Covers the four rendering branches of ``_format_check_summary`` (PASS /
PASS-with-skip_reason / FAIL / SKIP-with-skip_reason), the finding-total
accounting (only FAIL contributes), and the ``_summarize_findings``
>5-finding truncation.
"""

from __future__ import annotations

from pathlib import Path

import pedigree_summary as ps


def test_format_check_summary_renders_all_status_combinations():
    """Each status / skip_reason combination renders its own line shape."""
    results = [
        ps.CheckResult(name="required_columns", status="PASS"),
        ps.CheckResult(
            name="sex_role_consistency",
            status="PASS",
            count=1,
            skip_reason="1 overridden from role",
        ),
        ps.CheckResult(name="duplicate_ids", status="FAIL", count=3),
        ps.CheckResult(
            name="negative_ids",
            status="SKIP",
            skip_reason="id_dtype failed",
        ),
    ]
    out = ps._format_check_summary(Path("p.tsv"), n_total=5, results=results)
    # PASS line carries no parenthetical.
    assert "required columns present" in out
    assert " PASS\n" in out  # plain PASS row exists somewhere
    # PASS-with-count (0.9 sex_role_consistency override): trailing reason.
    assert "PASS (1 overridden from role)" in out
    # FAIL line carries the count.
    assert "no duplicate IDs " in out
    assert "FAIL (3)" in out
    # SKIP line carries the skip_reason.
    assert "SKIP (id_dtype failed)" in out


def test_format_check_summary_total_findings_only_counts_fail():
    """``result: N finding(s)`` only counts FAIL rows, not PASS-with-count."""
    results = [
        ps.CheckResult(name="duplicate_ids", status="FAIL", count=2),
        ps.CheckResult(
            name="sex_role_consistency",
            status="PASS",
            count=1,
            skip_reason="1 overridden from role",
        ),
        ps.CheckResult(
            name="acyclic",
            status="SKIP",
            skip_reason="parent dtype failed",
        ),
    ]
    out = ps._format_check_summary(Path("p.tsv"), n_total=5, results=results)
    assert "result: 2 finding(s)" in out


def test_summarize_findings_truncates_above_five():
    """8 findings → sample shows first 5 plus ``(and 3 more)`` suffix."""
    findings = [ps.Finding(check="negative_ids", id=i, row=i - 1, detail=f"id={i}") for i in range(1, 9)]
    out = ps._summarize_findings(findings)
    assert out.startswith("negative_ids: 8 finding(s) — ")
    # First five rows present.
    for i in range(1, 6):
        assert f"row {i - 1} (id={i})" in out
    # Truncation suffix.
    assert "(and 3 more)" in out
    # Findings past the fifth are NOT spelled out individually.
    assert "row 5 (id=6)" not in out
