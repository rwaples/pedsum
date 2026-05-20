"""Direct unit tests for ``_build_added_founders``.

Pins the four-way branch matrix (mother-only / father-only / both-roles
without no_sex_check / both-roles with no_sex_check) plus output sort
order and the >5-row reason-listing truncation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pedigree_summary as ps


def test_mother_only_missing_synthesises_female():
    """Missing id referenced only as mother → synthesised row with sex=F."""
    mothers = np.array([99, -1, -1], dtype=np.int64)
    fathers = np.array([-1, -1, -1], dtype=np.int64)
    id_index = pd.Index([1, 2, 3])
    out = ps._build_added_founders(mothers, fathers, id_index, no_sex_check=False)
    assert out == [{"id": 99, "sex": "F", "reason": "referenced as mother in row(s) [0]"}]


def test_father_only_missing_synthesises_male():
    """Missing id referenced only as father → synthesised row with sex=M."""
    mothers = np.array([-1, -1, -1], dtype=np.int64)
    fathers = np.array([88, -1, -1], dtype=np.int64)
    id_index = pd.Index([1, 2, 3])
    out = ps._build_added_founders(mothers, fathers, id_index, no_sex_check=False)
    assert out == [{"id": 88, "sex": "M", "reason": "referenced as father in row(s) [0]"}]


def test_both_roles_missing_blocked_without_no_sex_check():
    """Missing id used as both mother and father → no founder synthesised.

    Without --no-sex-check, _build_added_founders omits the conflicting
    row; validate surfaces it via parent_refs_sex_conflict instead.
    """
    mothers = np.array([99, -1, -1], dtype=np.int64)
    fathers = np.array([-1, 99, -1], dtype=np.int64)
    id_index = pd.Index([1, 2, 3])
    out = ps._build_added_founders(mothers, fathers, id_index, no_sex_check=False)
    assert out == []


def test_both_roles_missing_with_no_sex_check_synthesises_female():
    """With --no-sex-check, the both-roles fallback assigns sex=F."""
    mothers = np.array([99, -1, -1], dtype=np.int64)
    fathers = np.array([-1, 99, -1], dtype=np.int64)
    id_index = pd.Index([1, 2, 3])
    out = ps._build_added_founders(mothers, fathers, id_index, no_sex_check=True)
    assert len(out) == 1
    row = out[0]
    assert row["id"] == 99
    assert row["sex"] == "F"
    assert row["reason"].startswith("--no-sex-check; conflicting roles")
    assert "mother row(s) [0]" in row["reason"]
    assert "father row(s) [1]" in row["reason"]


def test_output_sorted_by_id():
    """Synthesised founder rows are sorted by id regardless of role mix."""
    # mother-only: 55, 10  /  father-only: 30  →  sorted output: 10, 30, 55.
    mothers = np.array([55, 10, -1, -1], dtype=np.int64)
    fathers = np.array([-1, -1, 30, -1], dtype=np.int64)
    id_index = pd.Index([1, 2, 3, 4])
    out = ps._build_added_founders(mothers, fathers, id_index, no_sex_check=False)
    assert [r["id"] for r in out] == [10, 30, 55]


def test_more_than_five_rows_truncates_listing():
    """Row-listing in the reason string truncates beyond five rows."""
    # id 99 referenced as mother in 8 rows → reason ends with "(and 3 more)".
    mothers = np.array([99] * 8, dtype=np.int64)
    fathers = np.full(8, -1, dtype=np.int64)
    id_index = pd.Index([1, 2, 3])
    out = ps._build_added_founders(mothers, fathers, id_index, no_sex_check=False)
    assert len(out) == 1
    assert out[0]["reason"].endswith("(and 3 more)")
    # First five rows are listed.
    assert "[0, 1, 2, 3, 4]" in out[0]["reason"]
