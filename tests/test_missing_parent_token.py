"""A parent written ``0``: ``zero_parent_present`` and ``--missing-parent-token`` (ADR 0006)."""

from __future__ import annotations

import polars as pl
import pytest
from conftest import load_validate_tsv_gz, run_check, run_pedsum, write_ped

from pedsum.base import PedigreeError
from pedsum.ids import parse_ids
from pedsum.validate import ValidationConfig

# The usual R / PLINK founder: both parents written 0, and no row has ID 0.
ZERO_FOUNDERS = [
    {"id": 1, "sex": "M", "mother": 0, "father": 0},
    {"id": 2, "sex": "F", "mother": 0, "father": 0},
    {"id": 3, "sex": "F", "mother": 2, "father": 1},
]


def _validate(tmp_path, rows, *extra):
    ped = write_ped(tmp_path / "p.tsv", rows)
    out = tmp_path / "out"
    return run_pedsum(["validate", "--in", str(ped), "--out", str(out), *extra]), out


def test_absent_zero_parent_blocks_and_names_the_flag(tmp_path):
    """Founders written ``0 0`` BLOCK plain validate, and the BLOCKED line names the fix."""
    r, out = _validate(tmp_path, ZERO_FOUNDERS)
    assert r.returncode == 2, r.stderr
    assert "parent ID 0 has no row; if the file writes a missing parent as 0, pass --missing-parent-token 0" in (
        r.stderr
    )
    assert not (out / "validate.tsv.gz").exists()
    log = pl.read_csv(out / "validate.log", separator="\t", infer_schema=False)
    zero = log.filter(pl.col("check") == "zero_parent_present")
    assert zero["id"].to_list() == ["0"]
    assert zero["detail"].item().startswith("id=0 referenced as mother in row(s) [0, 1] and as father in row(s) [0, 1]")


def test_drop_offending_no_longer_removes_zero_founders(tmp_path):
    """The Check is ``block``: --drop-offending refuses instead of deleting every founder written ``0 0``."""
    r, out = _validate(tmp_path, ZERO_FOUNDERS, "--drop-offending")
    assert r.returncode == 2, r.stderr
    assert "--drop-offending cannot fix these" in r.stderr
    assert not (out / "validate.dropped.tsv").exists()


def test_absent_zero_in_one_role_blocks(tmp_path):
    """A father written ``0`` with no row blocks too, instead of becoming an added founder named ``0``."""
    rows = [
        {"id": 1, "sex": "F", "mother": -1, "father": -1},
        {"id": 2, "sex": "M", "mother": 1, "father": 0},
    ]
    result = run_check("zero_parent_present", rows)
    assert result.status == "FAIL"
    assert "as father in row(s) [1]" in result.findings[0].detail


def test_zero_with_a_row_is_an_ordinary_id():
    """When ID 0 has a row, parents written ``0`` name it, as in simACE's pedigrees."""
    rows = [
        {"id": 0, "sex": "M", "mother": -1, "father": -1},
        {"id": 1, "sex": "F", "mother": -1, "father": -1},
        {"id": 2, "sex": "F", "mother": 1, "father": 0},
    ]
    assert run_check("zero_parent_present", rows).status == "PASS"


def test_string_ids_zero_parent():
    """String-mode IDs: a parent written ``0`` with no row FAILs; ``00`` is another ID."""
    rows = [
        {"id": "a", "sex": "M", "mother": "0", "father": "0"},
        {"id": "b", "sex": "F", "mother": "00", "father": "a"},
    ]
    result = run_check("zero_parent_present", rows)
    assert result.status == "FAIL"
    assert result.count == 1


def test_token_reads_zero_as_missing(tmp_path):
    """--missing-parent-token 0 reads the ``0`` parents as missing; validate writes them blank."""
    r, out = _validate(tmp_path, ZERO_FOUNDERS, "--missing-parent-token", "0")
    assert r.returncode == 0, r.stderr
    written = load_validate_tsv_gz(out)
    assert written["mother"].to_list() == [None, None, "2"]
    assert written["father"].to_list() == [None, None, "1"]
    # The Fixed Pedigree reads the same without the flag.
    s = run_pedsum(["summarize", "--in", str(out / "validate.tsv.gz"), "--out", str(tmp_path / "s")])
    assert s.returncode == 0, s.stderr


def test_every_command_takes_the_token(tmp_path):
    """The raw input summarizes with the flag and fails without it, as it validates."""
    ped = write_ped(tmp_path / "p.tsv", ZERO_FOUNDERS)
    r = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "s"), "--missing-parent-token", "0"])
    assert r.returncode == 0, r.stderr
    r = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "s2")])
    assert r.returncode == 1
    assert "zero_parent_present" in r.stderr


def test_token_that_is_an_id_blocks(tmp_path):
    """A token that is also the ID of a row cannot mean a missing parent: validate exits 2."""
    rows = [{"id": 0, "sex": "M", "mother": -1, "father": -1}, *ZERO_FOUNDERS]
    r, out = _validate(tmp_path, rows, "--missing-parent-token", "0")
    assert r.returncode == 2, r.stderr
    log = (out / "validate.log").read_text()
    assert "--missing-parent-token '0' is also the id of row(s) [0]" in log


@pytest.mark.parametrize(("token", "mother"), [("-9", "-9"), ("x", "x"), ("0", " 0 ")])
@pytest.mark.parametrize("ids", [["1", "2"], ["a", "b"]])
def test_parse_ids_token(token, mother, ids):
    """A token is missing in either ID mode, matched after stripping; integers like -9 included.

    A non-integer token puts no file in string mode, since it is not an ID.
    """
    _, mothers, fathers, labels = parse_ids(
        pl.Series("id", ids),
        pl.Series("mother", [mother, ids[0]]),
        pl.Series("father", ["-1", token]),
        missing_parent_tokens=(token,),
    )
    assert mothers[0] == -1
    assert fathers[1] == -1
    assert labels.is_string is (ids[0] == "a")


def test_parse_ids_token_matches_exactly():
    """Only the token itself is missing: ``00`` and ``X`` are IDs when the token is ``0`` / ``x``."""
    _, mothers, _, labels = parse_ids(
        pl.Series("id", ["a"]),
        pl.Series("mother", ["X"]),
        pl.Series("father", ["00"]),
        missing_parent_tokens=("x", "0"),
    )
    assert labels.label(mothers[0]) == "X"


def test_parse_ids_refuses_token_that_is_an_id():
    """parse_ids raises when a row's ID is a token."""
    with pytest.raises(PedigreeError, match="cannot be an individual"):
        parse_ids(
            pl.Series("id", ["0", "1"]),
            pl.Series("mother", ["-1", "0"]),
            pl.Series("father", ["-1", "-1"]),
            missing_parent_tokens=("0",),
        )


def test_parent_token_range_names_the_flag():
    """A parent below -1 FAILs with a detail that names --missing-parent-token."""
    rows = [{"id": 1, "sex": "M", "mother": -9, "father": -1}]
    result = run_check("parent_token_range_mother", rows)
    assert result.status == "FAIL"
    assert "--missing-parent-token -9" in result.findings[0].detail
    passed = run_check("parent_token_range_mother", rows, ValidationConfig(missing_parent_tokens=("-9",)))
    assert passed.status == "PASS"


def test_parent_refs_present_hint_names_the_flag():
    """The missing-row Finding for ID 0 names the flag, not 'convert to -1'."""
    rows = [
        {"id": 1, "sex": "F", "mother": -1, "father": -1},
        {"id": 2, "sex": "M", "mother": 1, "father": 0},
    ]
    detail = run_check("parent_refs_present_father", rows).findings[0].detail
    assert "pass --missing-parent-token 0" in detail
    assert "convert to -1" not in detail


def test_token_is_stripped():
    """A token passed with surrounding whitespace matches the stripped cells."""
    from pedsum.cli.parser import parse_args

    args = parse_args(["validate", "--in", "p.tsv", "--out", "o", "--missing-parent-token", " 0 "])
    config = ValidationConfig.from_args(args)
    assert config.missing_parent_tokens == ("0",)
    assert run_check("zero_parent_present", ZERO_FOUNDERS, config).status == "PASS"


def test_phantom_never_takes_a_token_id(tmp_path):
    """--fill-half-founders numbers phantoms past an integer token, so the output reads back with the flag."""
    rows = [
        {"id": 1, "sex": "M", "mother": 4, "father": 4},
        {"id": 2, "sex": "F", "mother": 4, "father": 4},
        {"id": 3, "sex": "F", "mother": 2, "father": 4},
    ]
    r, out = _validate(tmp_path, rows, "--missing-parent-token", "4", "--fill-half-founders")
    assert r.returncode in (0, 1), r.stderr
    written = load_validate_tsv_gz(out)
    assert "4" not in written["id"].to_list()
    s = run_pedsum(
        ["summarize", "--in", str(out / "validate.tsv.gz"), "--out", str(tmp_path / "s"), "--missing-parent-token", "4"]
    )
    assert s.returncode == 0, s.stderr
