"""``--min-parent-age``: ``birth_year_topology`` wants ``child - parent >= N`` years (ADR 0006)."""

from __future__ import annotations

import pytest
from conftest import run_check, run_pedsum
from conftest import write_ped as _write_ped

from pedsum.validate import ValidationConfig

ROWS = [
    {"id": "1", "sex": "M", "mother": "-1", "father": "-1", "by": "1950"},
    {"id": "2", "sex": "F", "mother": "-1", "father": "-1", "by": "1960"},
    {"id": "3", "sex": "F", "mother": "2", "father": "1", "by": "1962"},  # mother 2 years older
]


def _topology(min_age: int):
    return run_check("birth_year_topology", ROWS, ValidationConfig(birth_year_col="by", min_parent_age=min_age))


@pytest.mark.parametrize(("min_age", "flagged"), [(0, []), (2, []), (3, ["mother"]), (13, ["mother", "father"])])
def test_threshold_is_inclusive(min_age, flagged):
    """A parent exactly N years older passes; one year less FAILs, on each parent edge."""
    result = _topology(min_age)
    assert [("mother" if "mother" in f.detail else "father") for f in result.findings] == flagged


def test_detail_names_the_gap_and_the_flag():
    """A Finding under the age floor says how many years apart and names the flag."""
    (finding,) = _topology(3).findings
    assert finding.detail == (
        "id=3 birth_year=1962 is 2 year(s) after mother (id=2) birth_year=1960; --min-parent-age is 3"
    )
    assert finding.row == 2


def test_child_before_parent_keeps_its_wording():
    """A child born before its parent reads as before, whatever the floor."""
    rows = [{**ROWS[0]}, {**ROWS[1]}, {**ROWS[2], "by": "1955"}]
    details = [
        f.detail
        for f in run_check(
            "birth_year_topology", rows, ValidationConfig(birth_year_col="by", min_parent_age=3)
        ).findings
    ]
    assert "id=3 birth_year=1955 < mother (id=2) birth_year=1960" in details


def test_pass_notes_the_floor():
    """A PASS under a floor says which, so the summary shows the rule that ran."""
    assert _topology(2).note == "--min-parent-age 2"
    assert _topology(0).note is None


def test_cli_flag(tmp_path):
    """--min-parent-age reaches validate; a negative value is a usage error."""
    ped = _write_ped(tmp_path / "p.tsv", ROWS)
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "o"), "--birth-year-col", "by"])
    assert r.returncode == 0, r.stderr
    r = run_pedsum(
        ["validate", "--in", str(ped), "--out", str(tmp_path / "o"), "--birth-year-col", "by", "--min-parent-age", "3"]
    )
    assert r.returncode == 1, r.stderr  # keep_or_drop: written with the Finding
    assert "--min-parent-age is 3" in (tmp_path / "o" / "validate.log").read_text()
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "o"), "--min-parent-age", "-1"])
    assert r.returncode == 2
