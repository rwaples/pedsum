"""Exact duplicate rows collapse (``duplicate_rows``); IDs on rows that differ stay ``duplicate_ids`` (ADR 0006)."""

from __future__ import annotations

import polars as pl
import pytest
from conftest import load_validate_tsv_gz, run_check, run_pedsum

from pedsum.validate import ValidationConfig, ValidationContext, run_checks


def _write(path, text):
    path.write_text(text)
    return path


def _log(out):
    return pl.read_csv(out / "validate.log", separator="\t", infer_schema=False)


def _validate(tmp_path, text, *extra):
    ped = _write(tmp_path / "p.tsv", text)
    out = tmp_path / "out"
    return run_pedsum(["validate", "--in", str(ped), "--out", str(out), *extra]), out


PED = "id\tsex\tmother\tfather\n1\tM\t-1\t-1\n2\tF\t-1\t-1\n2\tF\t-1\t-1\n3\tF\t2\t1\n"


def test_exact_copy_collapses_and_exits_1(tmp_path):
    """The copy is removed with one Finding; the individual and its offspring's link stay."""
    r, out = _validate(tmp_path, PED)
    assert r.returncode == 1, r.stderr
    written = load_validate_tsv_gz(out)
    assert written["id"].to_list() == ["1", "2", "3"]
    assert written.filter(pl.col("id") == "3")["mother"].item() == "2"
    log = _log(out)
    assert log["check"].to_list() == ["duplicate_rows"]
    assert log["row"].to_list() == ["2"]
    assert log["detail"].item() == "row 2 repeats row 1 (id=2); the copy is removed"


def test_drop_offending_keeps_one_copy(tmp_path):
    """Only an autofix FAIL: --drop-offending has nothing to drop and exits 0, as for a synthesized founder."""
    r, out = _validate(tmp_path, PED, "--drop-offending")
    assert r.returncode == 0, r.stderr
    assert load_validate_tsv_gz(out)["id"].to_list() == ["1", "2", "3"]


def test_rows_compared_as_read():
    """NA and . are one missing parent, M and 1 one sex, 1980 and 1980.0 one birth year."""
    rows = [
        {"id": "1", "sex": "M", "mother": "NA", "father": "-1", "birth_year": "1980", "trait": "a"},
        {"id": "1", "sex": "1", "mother": ".", "father": "", "birth_year": "1980.0", "trait": "a"},
    ]
    config = ValidationConfig(birth_year_col="birth_year")
    result = run_check("duplicate_rows", rows, config)
    assert result.status == "FAIL"
    assert [f.row for f in result.findings] == [1]
    assert run_check("duplicate_ids", rows, config).status == "PASS"


def test_birth_year_as_text_without_the_flag():
    """Without --birth-year-col the birth-year column is one more text column: 1980 and 1980.0 differ."""
    rows = [
        {"id": "1", "sex": "M", "mother": "-1", "father": "-1", "birth_year": "1980"},
        {"id": "1", "sex": "M", "mother": "-1", "father": "-1", "birth_year": "1980.0"},
    ]
    assert run_check("duplicate_rows", rows).status == "PASS"
    assert run_check("duplicate_ids", rows).status == "FAIL"


def test_rows_differing_in_a_trait_block(tmp_path):
    """A trait column that differs makes it a Duplicate ID: plain validate BLOCKs, nothing collapses."""
    text = "id\tsex\tmother\tfather\tdx\n1\tM\t-1\t-1\t0\n1\tM\t-1\t-1\t1\n"
    r, out = _validate(tmp_path, text)
    assert r.returncode == 2, r.stderr
    assert _log(out)["check"].to_list() == ["duplicate_ids"]


def test_copies_and_a_differing_row():
    """Three copies and a differing row: two copies collapse, and the ID is still a Duplicate ID."""
    rows = [{"id": "1", "sex": "M", "mother": "-1", "father": "-1"}] * 3
    rows = [*rows, {"id": "1", "sex": "F", "mother": "-1", "father": "-1"}]
    df = pl.DataFrame(rows, schema=dict.fromkeys(rows[0], pl.String))
    ctx = ValidationContext(df, ValidationConfig())
    results, _ = run_checks(ctx)
    assert [(f.row, f.detail) for f in results["duplicate_rows"].findings] == [
        (1, "row 1 repeats row 0 (id=1); the copy is removed"),
        (2, "row 2 repeats row 0 (id=1); the copy is removed"),
    ]
    assert results["duplicate_ids"].count == 1
    assert ctx.input_rows.tolist() == [0, 3]


def test_later_findings_name_input_rows(tmp_path):
    """A Finding on a row after a removed copy names that row as the input numbers it."""
    text = "id\tsex\tmother\tfather\tby\n1\tM\t-1\t-1\t1950\n1\tM\t-1\t-1\t1950\n2\tF\t-1\t-1\t1950\n3\tF\t2\t1\t1940\n"
    r, out = _validate(tmp_path, text, "--birth-year-col", "by")
    assert r.returncode == 1, r.stderr
    log = _log(out).filter(pl.col("check") == "birth_year_topology")
    assert set(log["row"].to_list()) == {"3"}


def test_summarize_refuses_a_raw_duplicate(tmp_path):
    """A raw input with a copy fails summarize (fail-fast on autofix FAILs); validate's output runs."""
    ped = _write(tmp_path / "p.tsv", PED)
    r = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "s")])
    assert r.returncode == 1
    assert "duplicate_rows" in r.stderr
    run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "v")])
    r = run_pedsum(["summarize", "--in", str(tmp_path / "v" / "validate.tsv.gz"), "--out", str(tmp_path / "s2")])
    assert r.returncode == 0, r.stderr


def test_n_counts_the_input_rows(tmp_path):
    """The summary's N and the drop fraction count the file's rows, copies included."""
    r, _ = _validate(tmp_path, PED + "9\tM\t9\t-1\n", "--drop-offending")
    assert "(N=5)" in r.stderr
    assert "1 row(s) of 5 (20.0%)" in r.stderr


@pytest.mark.parametrize("name", ["__row", "__id", "__col0"])
def test_input_column_named_like_a_key(tmp_path, name):
    """An extra input column named like an internal key column does not crash the comparison."""
    text = f"id\tsex\tmother\tfather\t{name}\n1\tM\t-1\t-1\tx\n1\tM\t-1\t-1\tx\n"
    r, out = _validate(tmp_path, text)
    assert r.returncode == 1, r.stderr
    assert _log(out)["check"].to_list() == ["duplicate_rows"]
