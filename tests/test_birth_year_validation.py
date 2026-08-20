"""Behavior tests for the birth-year validation checks."""

from __future__ import annotations

import gzip

import polars as pl
import pytest
from conftest import run_pedsum
from conftest import write_ped as _write_ped

import pedigree_summary as ps


def test_validate_without_birth_year_col_skips_all_three(tmp_path):
    """Without --birth-year-col, all three birth-year checks SKIP."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "M", "mother": 2, "father": 1},
        ],
    )
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "out")])
    assert r.returncode == 0, r.stderr
    assert "birth_year column parses as numeric ... SKIP" in r.stderr
    assert "birth years within sanity range ....... SKIP" in r.stderr
    assert "child birth_year >= parent birth_year . SKIP" in r.stderr


def test_birth_year_dtype_check_fails_for_garbage(tmp_path):
    """Non-numeric birth-year tokens FAIL birth_year_dtype."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "birth_year": "abc"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "birth_year": "1955"},
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "birth_year": "1980"},
        ],
    )
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(tmp_path / "out"),
            "--birth-year-col",
            "birth_year",
        ]
    )
    # Not a hard block — exit 1 because there are findings.
    assert r.returncode == 1, r.stderr
    assert "birth_year column parses as numeric ... FAIL" in r.stderr
    # Once dtype fails, range and topology SKIP.
    assert "SKIP (birth_year_dtype failed)" in r.stderr


def test_birth_year_range_check_fails_for_typo(tmp_path):
    """Out-of-range birth year (e.g. 19888) FAILs birth_year_range."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "birth_year": "19888"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "birth_year": "1955"},
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "birth_year": "2010"},
        ],
    )
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(tmp_path / "out"),
            "--birth-year-col",
            "birth_year",
        ]
    )
    assert r.returncode == 1, r.stderr
    assert "birth years within sanity range ....... FAIL" in r.stderr
    # The validate.log row references id=1 and birth_year=19888.
    log = (tmp_path / "out" / "validate.log").read_text()
    assert "birth_year_range" in log
    assert "19888" in log
    assert "id=1" in log


def test_birth_year_topology_check_fails_for_inversion(tmp_path):
    """Child born before parent FAILs birth_year_topology (structured Finding)."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "birth_year": "1990"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "birth_year": "1995"},
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "birth_year": "1980"},
        ],
    )
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(tmp_path / "out"),
            "--birth-year-col",
            "birth_year",
        ]
    )
    assert r.returncode == 1, r.stderr
    assert "Traceback" not in r.stderr
    assert "child birth_year >= parent birth_year . FAIL" in r.stderr
    log = (tmp_path / "out" / "validate.log").read_text()
    assert "birth_year_topology" in log
    # Both mother and father edges should be flagged.
    assert log.count("birth_year_topology") >= 2


def test_birth_year_range_bounds_configurable(tmp_path):
    """--birth-year-min/--birth-year-max override the default range."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "birth_year": "1900"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "birth_year": "1905"},
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "birth_year": "1930"},
        ],
    )
    # Tight bounds → 1900 is below the floor.
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(tmp_path / "out"),
            "--birth-year-col",
            "birth_year",
            "--birth-year-min",
            "1910",
            "--birth-year-max",
            "1950",
        ]
    )
    assert r.returncode == 1, r.stderr
    assert "birth years within sanity range ....... FAIL" in r.stderr


def test_load_and_validate_raises_on_topology_violation(tmp_path):
    """load_and_validate raises PedigreeError (not ValueError) on topology."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "birth_year": "1990"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "birth_year": "1995"},
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "birth_year": "1980"},
        ],
    )
    with pytest.raises(ps.PedigreeError, match="birth_year_topology"):
        ps.load_and_validate(ped, birth_year_col="birth_year")


def test_summarize_surfaces_range_violation_cleanly(tmp_path):
    """Summarize emits a clean PedigreeError instead of a ValueError traceback."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "birth_year": "19888"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "birth_year": "1955"},
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "birth_year": "1980"},
        ],
    )
    r = run_pedsum(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(tmp_path / "s"),
            "--birth-year-col",
            "birth_year",
        ]
    )
    assert r.returncode == 1, r.stderr
    assert "Traceback" not in r.stderr
    assert "birth_year_range" in r.stderr


def test_birth_year_in_skip_section_when_column_missing(tmp_path):
    """If --birth-year-col names a column not in the file, the three checks SKIP cleanly."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "M", "mother": 2, "father": 1},
        ],
    )
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(tmp_path / "out"),
            "--birth-year-col",
            "birth_year",  # column doesn't exist
        ]
    )
    # Column absent from file → all three checks SKIP with a clear reason
    # (the required-columns check does not run for birth_year_col).
    assert r.returncode == 0, r.stderr
    assert "Birth years" in r.stderr
    assert "not present in file" in r.stderr


def test_birth_year_fixed_output_includes_topological_reorder(tmp_path):
    """Birth-year column is preserved in .validate.tsv.gz even after reorder."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            # Unordered: child before parents.
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "birth_year": "1980"},
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "birth_year": "1950"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "birth_year": "1955"},
        ],
    )
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(tmp_path / "out"),
            "--birth-year-col",
            "birth_year",
        ]
    )
    assert r.returncode == 0, r.stderr
    with gzip.open(tmp_path / "out" / "validate.tsv.gz", "rb") as fh:
        fixed = pl.read_csv(fh.read(), separator="\t", infer_schema=False)
    assert "birth_year" in fixed.columns
    ids_in_order = fixed["id"].cast(pl.Int64).to_list()
    assert ids_in_order.index(1) < ids_in_order.index(3)
    assert ids_in_order.index(2) < ids_in_order.index(3)
