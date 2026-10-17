"""Behavior tests for --birth-date-col and --twin-max-days (pedsum#26)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import load_validate_tsv_gz, run_pedsum, write_ped

from pedsum.base import PedigreeError
from pedsum.fixed_pedigree import fix_pedigree, multiple_birth_sets
from pedsum.parse import as_birth_date_col
from pedsum.validate import ValidationConfig, ValidationContext, run_checks

FOUNDERS = [
    {"id": "1", "sex": "F", "mother": "", "father": "", "birth_year": "1960", "birth_date": "1960-03-01"},
    {"id": "2", "sex": "M", "mother": "", "father": "", "birth_year": "1960", "birth_date": "1960-05-01"},
    {"id": "3", "sex": "M", "mother": "", "father": "", "birth_year": "1960", "birth_date": "1960-07-01"},
]


def _child(id_: str, mother: str, father: str, date: str, year: str | None = None) -> dict:
    year = year if year is not None else (date[:4] if date[:1].isdigit() else "")
    return {"id": id_, "sex": "F", "mother": mother, "father": father, "birth_year": year, "birth_date": date}


def _sets(rows: list[dict], max_days: int) -> dict[str, int]:
    """Each set member's set number in the Fixed Pedigree of ``rows``, keyed by ID."""
    df = pl.DataFrame(rows, schema=dict.fromkeys(rows[0], pl.String))
    ctx = ValidationContext(df, ValidationConfig(birth_date_col="birth_date", twin_max_days=max_days))
    run_checks(ctx)
    fp = fix_pedigree(ctx)
    assert fp.multiple_birth_set is not None
    return {fp.labels.label(i): int(s) for i, s in zip(fp.ids, fp.multiple_birth_set, strict=True) if s >= 0}


def test_birth_date_parses_iso_and_missing_tokens():
    """ISO dates parse to datetime64[D]; missing tokens are NaT."""
    got = as_birth_date_col(pl.Series(["2001-01-05", "NA", "", None, " 1999-12-31 "]), "bd")
    assert got.dtype == np.dtype("datetime64[D]")
    assert got[0] == np.datetime64("2001-01-05")
    assert np.isnat(got[1:4]).all()
    assert got[4] == np.datetime64("1999-12-31")


@pytest.mark.parametrize("token", ["2001-1-5", "2001-02-30", "05.01.2001", "20010105", "2001-01-05T00:00", "1988"])
def test_birth_date_rejects_non_iso(token):
    """Only zero-padded ISO dates of real calendar days parse."""
    with pytest.raises(PedigreeError, match="ISO dates"):
        as_birth_date_col(pl.Series([token]), "bd")


def test_full_sibs_within_the_limit_form_numbered_sets():
    """Twins and a triplet chained over midnight are sets; a later sib and a sib of unknown date are not."""
    rows = [
        *FOUNDERS,
        _child("10", "1", "2", "1985-06-01"),
        _child("11", "1", "2", "1985-06-02"),
        _child("12", "1", "2", "1987-01-01"),
        _child("13", "1", "2", "1990-02-10"),
        _child("14", "1", "2", "1990-02-10"),
        _child("15", "1", "2", "1990-02-11"),
        _child("16", "1", "2", "NA"),
    ]
    assert _sets(rows, max_days=1) == {"10": 1, "11": 1, "13": 2, "14": 2, "15": 2}


def test_zero_days_needs_the_same_date():
    """--twin-max-days 0 groups only same-day births."""
    rows = [*FOUNDERS, _child("10", "1", "2", "1985-06-01"), _child("11", "1", "2", "1985-06-02")]
    assert _sets(rows, max_days=0) == {}
    assert _sets(rows, max_days=1) == {"10": 1, "11": 1}


def test_half_sibs_born_the_same_day_are_not_a_set():
    """A set needs both parents shared, and both known."""
    rows = [
        *FOUNDERS,
        _child("10", "1", "2", "1985-06-01"),
        _child("11", "1", "3", "1985-06-01"),
        _child("12", "1", "", "1985-06-01"),
        _child("13", "1", "", "1985-06-01"),
    ]
    assert _sets(rows, max_days=1) == {}


def test_parent_without_a_row_still_groups_its_offspring():
    """Sibships are grouped on parent IDs, so a father pedsum adds as a founder still counts."""
    rows = [*FOUNDERS, _child("10", "1", "99", "1985-06-01"), _child("11", "1", "99", "1985-06-01")]
    assert _sets(rows, max_days=0) == {"10": 1, "11": 1}


def test_chain_wider_than_the_limit_warns(caplog):
    """A set that a chain of births stretches past max_days is kept, with a WARNING."""
    dates = np.array(["1990-06-01", "1990-06-02", "1990-06-03"], dtype="datetime64[D]")
    parents = np.array([1, 1, 1], dtype=np.int64)
    with caplog.at_level("WARNING"):
        sets = multiple_birth_sets(parents, parents + 1, dates, max_days=1)
    assert sets.tolist() == [1, 1, 1]
    assert "1 multiple-birth set(s) span more than --twin-max-days 1" in caplog.text
    assert "(widest 2 days)" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        multiple_birth_sets(parents, parents + 1, dates, max_days=2)
    assert caplog.text == ""


def _validate(tmp_path, rows, *flags, out="out"):
    ped = write_ped(tmp_path / "p.tsv", rows)
    return run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / out), *flags])


def test_off_by_default_even_with_a_birth_date_column(tmp_path):
    """Without --twin-max-days no column is added and no multiple-births line is printed."""
    rows = [*FOUNDERS, _child("10", "1", "2", "1985-06-01"), _child("11", "1", "2", "1985-06-01")]
    r = _validate(tmp_path, rows, "--birth-date-col", "birth_date")
    assert r.returncode == 0, r.stderr
    assert "multiple births" not in r.stderr
    assert "multiple_birth_set" not in load_validate_tsv_gz(tmp_path / "out").columns


def test_twin_max_days_needs_birth_date_col(tmp_path):
    """--twin-max-days without --birth-date-col is a usage error."""
    r = _validate(tmp_path, FOUNDERS, "--twin-max-days", "1")
    assert r.returncode == 2
    assert "--twin-max-days needs --birth-date-col" in r.stderr


def test_twin_max_days_requires_the_column_to_exist(tmp_path):
    """With --twin-max-days, a birth-date column missing from the file BLOCKs."""
    r = _validate(tmp_path, FOUNDERS, "--birth-date-col", "dob", "--twin-max-days", "1")
    assert r.returncode == 2, r.stderr
    assert "BLOCKED" in r.stderr
    assert not (tmp_path / "out" / "validate.tsv.gz").exists()


def test_unparsable_birth_date_blocks(tmp_path):
    """birth_date_dtype is a BLOCK Check: no pedigree with an unparsable date."""
    rows = [*FOUNDERS, _child("10", "1", "2", "06/01/1985", year="1985")]
    r = _validate(tmp_path, rows, "--birth-date-col", "birth_date")
    assert r.returncode == 2, r.stderr
    assert "birth_date column parses as ISO dates . FAIL" in r.stderr
    assert not (tmp_path / "out" / "validate.tsv.gz").exists()


def test_column_written_with_founders_blank(tmp_path):
    """validate.tsv.gz gets multiple_birth_set: set numbers for members, blank otherwise and on added founders."""
    rows = [
        *FOUNDERS,
        _child("10", "1", "2", "1985-06-01"),
        _child("11", "1", "2", "1985-06-01"),
        _child("12", "1", "2", "1987-01-01"),
        _child("13", "1", "99", "1990-01-01"),
        _child("14", "1", "99", "1990-01-01"),
    ]
    r = _validate(tmp_path, rows, "--birth-date-col", "birth_date", "--twin-max-days", "1")
    assert r.returncode == 1, r.stderr  # parent_refs_present_father adds founder 99
    assert "multiple births: 2 set(s), 4 individual(s) (--twin-max-days 1)" in r.stderr
    out = load_validate_tsv_gz(tmp_path / "out")
    got = dict(zip(out["id"].to_list(), out["multiple_birth_set"].to_list(), strict=True))
    assert got == {"99": None, "1": None, "2": None, "3": None, "10": "1", "11": "1", "12": None, "13": "2", "14": "2"}


def test_revalidating_replaces_the_column(tmp_path):
    """Validating validate.tsv.gz again with another limit rewrites multiple_birth_set, not a second copy."""
    rows = [*FOUNDERS, _child("10", "1", "2", "1985-06-01"), _child("11", "1", "2", "1985-06-03")]
    r = _validate(tmp_path, rows, "--birth-date-col", "birth_date", "--twin-max-days", "2")
    assert r.returncode == 0, r.stderr
    assert load_validate_tsv_gz(tmp_path / "out")["multiple_birth_set"].to_list()[-2:] == ["1", "1"]
    again = tmp_path / "again"
    r = run_pedsum(
        [
            "validate", "--in", str(tmp_path / "out" / "validate.tsv.gz"), "--out", str(again),
            "--birth-date-col", "birth_date", "--twin-max-days", "1",
        ]
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    out = load_validate_tsv_gz(again)
    assert out.columns.count("multiple_birth_set") == 1
    assert out["multiple_birth_set"].to_list()[-2:] == [None, None]


def test_revalidating_without_the_flag_drops_the_column(tmp_path):
    """Without --twin-max-days, an input multiple_birth_set column is dropped, not passed on stale."""
    rows = [*FOUNDERS, _child("10", "1", "2", "1985-06-01"), _child("11", "1", "2", "1985-06-01")]
    r = _validate(tmp_path, rows, "--birth-date-col", "birth_date", "--twin-max-days", "0")
    assert r.returncode == 0, r.stderr
    again = tmp_path / "again"
    r = run_pedsum(["validate", "--in", str(tmp_path / "out" / "validate.tsv.gz"), "--out", str(again)])
    assert r.returncode == 0, r.stderr
    assert "dropped the input's multiple_birth_set column" in r.stderr
    assert "multiple_birth_set" not in load_validate_tsv_gz(again).columns


def test_birth_date_outside_birth_year_is_kept_or_dropped(tmp_path):
    """birth_date_year_consistency is KEEP_OR_DROP: plain validate keeps the row, --drop-offending removes it."""
    rows = [*FOUNDERS, _child("10", "1", "2", "1985-06-01", year="1986"), _child("11", "1", "2", "1987-01-01")]
    flags = ("--birth-year-col", "birth_year", "--birth-date-col", "birth_date")
    r = _validate(tmp_path, rows, *flags)
    assert r.returncode == 1, r.stderr
    assert "birth dates fall in their birth year .. FAIL (1)" in r.stderr
    log = (tmp_path / "out" / "validate.log").read_text()
    assert "id=10 birth_date=1985-06-01 but birth_year=1986" in log
    assert "10" in load_validate_tsv_gz(tmp_path / "out")["id"].to_list()

    r = _validate(tmp_path, rows, *flags, "--drop-offending", out="dropped")
    assert r.returncode == 1, r.stderr
    assert "10" not in load_validate_tsv_gz(tmp_path / "dropped")["id"].to_list()


def test_birth_date_outside_the_year_bounds_is_flagged(tmp_path):
    """birth_date_range applies --birth-year-min/--birth-year-max to the date's year."""
    rows = [*FOUNDERS, _child("10", "1", "2", "2985-06-01", year=""), _child("11", "1", "2", "1987-01-01")]
    r = _validate(tmp_path, rows, "--birth-date-col", "birth_date", "--birth-year-max", "2030")
    assert r.returncode == 1, r.stderr
    assert "birth dates within sanity range ....... FAIL (1)" in r.stderr
    log = (tmp_path / "out" / "validate.log").read_text()
    assert "id=10 birth_date=2985-06-01 is outside the sanity range [1800, 2030]" in log


def test_year_consistency_skips_without_birth_year_col(tmp_path):
    """Without --birth-year-col there is no year to compare against."""
    rows = [*FOUNDERS, _child("10", "1", "2", "1985-06-01", year="1986")]
    r = _validate(tmp_path, rows, "--birth-date-col", "birth_date")
    assert r.returncode == 0, r.stderr
    assert "birth dates fall in their birth year .. SKIP (no --birth-year-col specified)" in r.stderr


def test_drop_offending_annotates_the_reduced_pedigree(tmp_path):
    """Under --drop-offending the sets and the printed count are the Reduced Pedigree's; the self-verify passes."""
    rows = [
        *FOUNDERS,
        _child("10", "1", "2", "1985-06-01"),
        _child("11", "1", "2", "1985-06-01"),
        _child("12", "1", "2", "1985-06-01", year="1990"),
    ]
    r = _validate(
        tmp_path,
        rows,
        "--birth-year-col", "birth_year", "--birth-date-col", "birth_date", "--twin-max-days", "0",
        "--drop-offending",
    )  # fmt: skip
    assert r.returncode == 1, r.stderr
    assert "multiple births: 1 set(s), 2 individual(s)" in r.stderr
    out = load_validate_tsv_gz(tmp_path / "out")
    got = dict(zip(out["id"].to_list(), out["multiple_birth_set"].to_list(), strict=True))
    assert got == {"1": None, "2": None, "3": None, "10": "1", "11": "1"}
