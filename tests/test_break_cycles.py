"""``validate --break-cycles``: clear the links the birth dates or years show to be wrong (ADR 0006, #28)."""

from __future__ import annotations

import polars as pl
from conftest import load_validate_tsv_gz, run_check, run_pedsum

from pedsum.validate import ValidationConfig, validate_pedigree

BREAK = ValidationConfig(birth_year_col="by", break_cycles=True)

# 6 and 7 are each other's mother; 7's mother link is the wrong one (6 is
# born after 7). 8 is 6's daughter, below the cycle.
CYCLE = "id\tsex\tmother\tfather\tby\n1\tM\t-1\t-1\t1950\n6\tF\t7\t1\t1990\n7\tF\t6\t1\t1980\n8\tF\t6\t1\t2010\n"


def _write(path, text):
    path.write_text(text)
    return path


def _validate(tmp_path, text, *extra):
    ped = _write(tmp_path / "p.tsv", text)
    out = tmp_path / "out"
    return run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--birth-year-col", "by", *extra]), out


def test_clears_the_late_parent_link(tmp_path):
    """The link to a parent born after the child is cleared; everyone stays; validate exits 1."""
    r, out = _validate(tmp_path, CYCLE, "--break-cycles")
    assert r.returncode == 1, r.stderr
    written = load_validate_tsv_gz(out)
    assert sorted(written["id"].to_list()) == ["1", "6", "7", "8"]
    assert written.filter(pl.col("id") == "7")["mother"].item() is None
    assert written.filter(pl.col("id") == "6")["mother"].item() == "7"
    cleared = pl.read_csv(out / "validate.cleared_links.tsv", separator="\t", infer_schema=False)
    assert cleared.rows() == [("7", "mother", "6", "in a cycle, mother born 1990 after the child (born 1980)")]
    log = pl.read_csv(out / "validate.log", separator="\t", infer_schema=False)
    assert log["check"].to_list() == ["cycle_breaks"]
    assert log["row"].to_list() == ["2"]


def test_without_the_flag_the_cycle_blocks(tmp_path):
    """The default is unchanged: the cycle BLOCKs and no manifest is written."""
    r, out = _validate(tmp_path, CYCLE)
    assert r.returncode == 2, r.stderr
    assert "cycle detected" in r.stderr
    assert not (out / "validate.cleared_links.tsv").exists()


def test_self_link_cleared():
    """A self-link is cleared whatever the birth years say."""
    rows = [
        {"id": "1", "sex": "M", "mother": "-1", "father": "-1", "by": "1950"},
        {"id": "2", "sex": "F", "mother": "-1", "father": "2", "by": "1960"},
    ]
    result = run_check("cycle_breaks", rows, BREAK)
    assert [f.detail for f in result.findings] == ["id=2: cleared father id=2, listed as own father"]
    assert run_check("self_loops", rows, BREAK).status == "PASS"
    assert run_check("acyclic", rows, BREAK).status == "PASS"


def test_cycle_without_evidence_is_left(tmp_path):
    """A cycle whose parents are not born after their children keeps its links and still BLOCKs."""
    text = CYCLE.replace("6\tF\t7\t1\t1990", "6\tF\t7\t1\t1980")  # both 1980
    r, _ = _validate(tmp_path, text, "--break-cycles")
    assert r.returncode == 2, r.stderr
    assert "cycle detected" in r.stderr


def test_unknown_birth_year_is_no_evidence():
    """A link whose parent or child has no birth year is never cleared."""
    rows = [
        {"id": "6", "sex": "F", "mother": "7", "father": "-1", "by": "-1"},
        {"id": "7", "sex": "F", "mother": "6", "father": "-1", "by": "1980"},
    ]
    assert run_check("cycle_breaks", rows, BREAK).status == "PASS"
    assert run_check("acyclic", rows, BREAK).status == "FAIL"


def test_late_parent_outside_a_cycle_is_kept():
    """A parent born after the child but in no cycle is birth_year_topology's, not cleared."""
    rows = [
        {"id": "1", "sex": "M", "mother": "-1", "father": "-1", "by": "2000"},
        {"id": "2", "sex": "F", "mother": "-1", "father": "1", "by": "1990"},
    ]
    assert run_check("cycle_breaks", rows, BREAK).status == "PASS"
    assert run_check("birth_year_topology", rows, BREAK).status == "FAIL"


def test_drop_offending_breaks_then_drops_what_is_left(tmp_path):
    """Under --drop-offending the cleared links stay cleared and only an unbroken cycle loses members."""
    text = CYCLE + "20\tF\t21\t1\t1980\n21\tF\t20\t1\t1980\n"
    r, out = _validate(tmp_path, text, "--break-cycles", "--drop-offending")
    assert r.returncode == 1, r.stderr
    written = load_validate_tsv_gz(out)
    assert sorted(written["id"].to_list()) == ["1", "6", "7", "8"]
    cleared = pl.read_csv(out / "validate.cleared_links.tsv", separator="\t", infer_schema=False)
    assert cleared["id"].to_list() == ["7"]
    dropped = pl.read_csv(out / "validate.dropped.tsv", separator="\t", infer_schema=False)
    assert sorted(dropped["id"].to_list()) == ["20", "21"]


def test_needs_birth_year_or_birth_date_col(tmp_path):
    """--break-cycles with neither --birth-year-col nor --birth-date-col is a usage error."""
    ped = _write(tmp_path / "p.tsv", CYCLE)
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "o"), "--break-cycles"])
    assert r.returncode == 2
    assert "--break-cycles needs --birth-year-col or --birth-date-col" in r.stderr


def test_missing_birth_year_column_blocks(tmp_path):
    """A named birth-year column the file lacks BLOCKs instead of skipping the clearing."""
    r, out = _validate(tmp_path, CYCLE.replace("\tby\n", "\tyear\n", 1), "--break-cycles")
    assert r.returncode == 2, r.stderr
    assert "required_columns" in (out / "validate.log").read_text()


def test_stale_manifest_removed(tmp_path):
    """A later run without --break-cycles removes the manifest an earlier run left."""
    _, out = _validate(tmp_path, CYCLE, "--break-cycles")
    assert (out / "validate.cleared_links.tsv").exists()
    _validate(tmp_path, CYCLE)
    assert not (out / "validate.cleared_links.tsv").exists()


DATES = ValidationConfig(birth_date_col="bd", break_cycles=True)
BOTH = ValidationConfig(birth_year_col="by", birth_date_col="bd", break_cycles=True)
# CYCLE with birth dates instead of birth years.
DATED_CYCLE = (
    "id\tsex\tmother\tfather\tbd\n1\tM\t-1\t-1\t1950-01-01\n6\tF\t7\t1\t1990-01-01\n"
    "7\tF\t6\t1\t1980-01-01\n8\tF\t6\t1\t2010-01-01\n"
)


def _validate_dated(tmp_path, text):
    ped = _write(tmp_path / "p.tsv", text)
    out = tmp_path / "out"
    return run_pedsum(
        ["validate", "--in", str(ped), "--out", str(out), "--birth-date-col", "bd", "--break-cycles"]
    ), out


def test_birth_dates_alone_break_the_cycle(tmp_path):
    """With --birth-date-col and no --birth-year-col, the link to a parent born later by date is cleared (#28)."""
    r, out = _validate_dated(tmp_path, DATED_CYCLE)
    assert r.returncode == 1, r.stderr
    cleared = pl.read_csv(out / "validate.cleared_links.tsv", separator="\t", infer_schema=False)
    assert cleared.rows() == [
        ("7", "mother", "6", "in a cycle, mother born 1990-01-01 after the child (born 1980-01-01)")
    ]


def test_dates_resolve_a_parent_born_in_the_childs_year():
    """Birth years tie, so only the dates show the mother is born after the child (#28)."""
    rows = [
        {"id": "6", "sex": "F", "mother": "7", "father": "-1", "by": "1980", "bd": "1980-09-01"},
        {"id": "7", "sex": "F", "mother": "6", "father": "-1", "by": "1980", "bd": "1980-03-01"},
    ]
    result = run_check("cycle_breaks", rows, BOTH)
    assert [f.detail for f in result.findings] == [
        "id=7: cleared mother id=6, in a cycle, mother born 1980-09-01 after the child (born 1980-03-01)"
    ]


def test_birth_years_cover_links_without_both_dates():
    """A link missing a date falls back to the birth years (#28)."""
    rows = [
        {"id": "6", "sex": "F", "mother": "7", "father": "-1", "by": "1990", "bd": ""},
        {"id": "7", "sex": "F", "mother": "6", "father": "-1", "by": "1980", "bd": "1980-03-01"},
    ]
    result = run_check("cycle_breaks", rows, BOTH)
    assert [f.detail for f in result.findings] == [
        "id=7: cleared mother id=6, in a cycle, mother born 1990 after the child (born 1980)"
    ]
    assert run_check("cycle_breaks", rows, DATES).status == "PASS"


def test_missing_birth_date_column_blocks(tmp_path):
    """A named birth-date column the file lacks BLOCKs instead of skipping the clearing (#28)."""
    r, out = _validate_dated(tmp_path, DATED_CYCLE.replace("\tbd\n", "\tdate\n", 1))
    assert r.returncode == 2, r.stderr
    assert "required_columns" in (out / "validate.log").read_text()


def test_a_birth_column_that_does_not_parse_clears_nothing():
    """With a named column unparsed, cycle_breaks SKIPs instead of acting on the other one (#28)."""
    rows = [
        {"id": "6", "sex": "F", "mother": "7", "father": "-1", "by": "abc", "bd": "1990-01-01"},
        {"id": "7", "sex": "F", "mother": "6", "father": "-1", "by": "1980", "bd": "1980-01-01"},
    ]
    result = run_check("cycle_breaks", rows, BOTH)
    assert result.status == "SKIP"
    assert result.note == "--birth-year-col 'by' did not parse"


def test_a_date_on_one_side_and_a_year_on_the_other_are_compared():
    """A missing birth year is taken from the birth date, so a date and a year still order the link (#28)."""
    rows = [
        {"id": "6", "sex": "F", "mother": "7", "father": "-1", "by": "", "bd": "1990-05-01"},
        {"id": "7", "sex": "F", "mother": "6", "father": "-1", "by": "1980", "bd": ""},
    ]
    result = run_check("cycle_breaks", rows, BOTH)
    assert [f.detail for f in result.findings] == [
        "id=7: cleared mother id=6, in a cycle, mother born 1990 after the child (born 1980)"
    ]


def test_library_callers_get_the_named_columns_required(tmp_path):
    """Without the CLI, --break-cycles still needs every named birth column in the file (#28)."""
    ped = _write(tmp_path / "p.tsv", CYCLE.replace("\tby\n", "\tyear\n", 1))
    _, results, _, _ = validate_pedigree(ped, BREAK)
    assert {r.name: r.status for r in results}["required_columns"] == "FAIL"
