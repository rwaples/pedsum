"""``--min-parent-age``: ``birth_year_topology`` wants ``child - parent >= N`` years (ADR 0006)."""

from __future__ import annotations

import pytest
from conftest import run_check, run_pedsum
from conftest import write_ped as _write_ped

from pedsum.checks import CheckStatus
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


MIXED = [
    {"id": "1", "sex": "F", "mother": "-1", "father": "-1", "by": "", "bd": "2000-01-01"},
    {"id": "2", "sex": "M", "mother": "1", "father": "-1", "by": "1990", "bd": ""},
]
MIXED_CONFIG = ValidationConfig(birth_year_col="by", birth_date_col="bd")


def test_year_topology_takes_the_year_of_a_birth_date():
    """A mother known only by date and a child only by year: the year check compares 2000 with 1990."""
    result = run_check("birth_year_topology", MIXED, MIXED_CONFIG)
    assert [f.detail for f in result.findings] == [
        "id=2 birth_year=1990 < mother (id=1) birth_year=2000 (from birth_date)"
    ]
    assert run_check("birth_date_topology", MIXED, MIXED_CONFIG).findings == []


def test_an_edge_with_both_dates_is_checked_once():
    """Both dates known: birth_date_topology owns the edge, even when the years disagree with it."""
    rows = [{**MIXED[0], "by": "2000"}, {**MIXED[1], "bd": "1990-06-01"}]
    assert run_check("birth_year_topology", rows, MIXED_CONFIG).findings == []
    assert len(run_check("birth_date_topology", rows, MIXED_CONFIG).findings) == 1


OLD = [
    {"id": "1", "sex": "M", "mother": "-1", "father": "-1", "by": "1900", "bd": ""},
    {"id": "2", "sex": "F", "mother": "-1", "father": "-1", "by": "1950", "bd": "1950-06-01"},
    {"id": "3", "sex": "F", "mother": "2", "father": "1", "by": "1980", "bd": "1980-05-31"},
]


@pytest.mark.parametrize(("max_age", "year_flagged", "date_flagged"), [(0, 0, 0), (79, 0, 0), (78, 1, 0), (28, 1, 1)])
def test_max_parent_age(max_age, year_flagged, date_flagged):
    """A parent too old fails: the father on years (80, above N + 1), the mother on dates (29 completed years)."""
    config = ValidationConfig(birth_year_col="by", birth_date_col="bd", max_parent_age=max_age)
    assert len(run_check("birth_year_topology", OLD, config).findings) == year_flagged
    assert len(run_check("birth_date_topology", OLD, config).findings) == date_flagged


def test_max_parent_age_detail_and_note():
    """A Finding over the age ceiling names the gap and the flag; a PASS under it notes the flag."""
    config = ValidationConfig(birth_year_col="by", max_parent_age=70)
    result = run_check("birth_year_topology", OLD, config)
    assert [f.detail for f in result.findings] == [
        "id=3 birth_year=1980 is 80 year(s) after father (id=1) birth_year=1900; --max-parent-age is 70 (more than 71 on birth years)"
    ]
    rows = [OLD[1], {**OLD[2], "father": "-1"}]
    assert run_check("birth_year_topology", rows, config).note == "--max-parent-age 70"


def test_max_parent_age_cli_flag(tmp_path):
    """--max-parent-age reaches validate and summarize; a negative value is a usage error."""
    ped = _write_ped(tmp_path / "p.tsv", OLD)
    r = run_pedsum(
        ["validate", "--in", str(ped), "--out", str(tmp_path / "o"), "--birth-year-col", "by", "--max-parent-age", "70"]
    )
    assert r.returncode == 1, r.stderr
    assert "--max-parent-age is 70" in (tmp_path / "o" / "validate.log").read_text()
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "o"), "--max-parent-age", "-1"])
    assert r.returncode == 2
    r = run_pedsum(["summarize", "--help"])
    assert "--max-parent-age" in r.stdout


@pytest.mark.parametrize("command", ["validate", "summarize"])
def test_min_parent_age_above_max_parent_age_is_a_usage_error(tmp_path, command):
    """A floor above a set ceiling would flag every parent-child link; a ceiling of 0 is off and allows any floor."""
    ped = _write_ped(tmp_path / "p.tsv", ROWS)
    base = [command, "--in", str(ped), "--out", str(tmp_path / "o"), "--birth-year-col", "by"]
    r = run_pedsum([*base, "--min-parent-age", "50", "--max-parent-age", "40"])
    assert r.returncode == 2
    assert "--min-parent-age 50 is above --max-parent-age 40" in r.stderr
    r = run_pedsum([*base, "--min-parent-age", "40", "--max-parent-age", "40"])
    assert r.returncode != 2, r.stderr
    r = run_pedsum([*base, "--min-parent-age", "50"])
    assert r.returncode != 2, r.stderr


@pytest.mark.parametrize(("father_year", "flagged"), [("1950", 0), ("1949", 1)])
def test_max_parent_age_on_years_allows_the_year_a_difference_overstates(father_year, flagged):
    """Born Dec 1950, a father of a Jan 2021 child is 70: on years 71 passes --max-parent-age 70, and 72 fails."""
    rows = [
        {"id": "1", "sex": "M", "mother": "-1", "father": "-1", "by": father_year},
        {"id": "2", "sex": "F", "mother": "-1", "father": "1", "by": "2021"},
    ]
    config = ValidationConfig(birth_year_col="by", max_parent_age=70)
    assert len(run_check("birth_year_topology", rows, config).findings) == flagged


def test_a_failed_birth_date_column_leaves_every_edge_to_the_years():
    """birth_date_dtype FAILs: birth_year_topology still checks the dated edge on its years."""
    rows = [
        {"id": "1", "sex": "F", "mother": "-1", "father": "-1", "by": "2000", "bd": "2000-01-01"},
        {"id": "2", "sex": "F", "mother": "1", "father": "-1", "by": "1990", "bd": "not a date"},
    ]
    config = ValidationConfig(birth_year_col="by", birth_date_col="bd")
    assert run_check("birth_date_dtype", rows, config).status == CheckStatus.FAIL
    result = run_check("birth_year_topology", rows, config)
    assert [f.id for f in result.findings] == [2]
