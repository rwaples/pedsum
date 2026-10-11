"""The Fixed Pedigree through its interface: ``fix_pedigree``, ``render_fixed_pedigree`` and ``fix_validated`` (ADR 0005)."""

from __future__ import annotations

import numpy as np
from conftest import write_ped as _write_ped

from pedsum.base import SEX_FEMALE, SEX_MALE
from pedsum.fixed_pedigree import fix_pedigree, render_fixed_pedigree
from pedsum.validate import fix_validated, validate_pedigree

_COLS = {"id_col": "id", "sex_col": "sex", "mother_col": "mother", "father_col": "father"}


def _fixed(path, **kwargs):
    _n, _results, _findings, ctx = validate_pedigree(path)
    fp = fix_pedigree(ctx, **kwargs)
    return fp, render_fixed_pedigree(fp, ctx.df_raw, **_COLS)


def test_missing_parents_become_founder_rows_written_first(tmp_path):
    """Each missing parent ID gets a founder row, by ID, ahead of the input rows."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 3, "sex": "M", "mother": 1, "father": 2},
            {"id": 4, "sex": "F", "mother": 1, "father": 2},
        ],
    )
    fp, out = _fixed(ped)
    assert [(f["id"], f["sex"]) for f in fp.founders] == [(1, SEX_FEMALE), (2, SEX_MALE)]
    assert out["id"].to_list() == ["1", "2", "3", "4"]
    assert out["mother"].to_list() == ["-1", "-1", "1", "1"]
    assert out["sex"].to_list() == ["2", "1", "1", "2"]  # PLINK coding throughout
    assert out["sex_source"].to_list() == [None, None, "input", "input"]


def test_rows_go_parents_first_and_keep_extra_columns(tmp_path):
    """Input rows are reordered parents-first and carry their own extra-column tokens."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 3, "sex": "F", "mother": 2, "father": 1, "note": "child"},
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "note": "dad"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "note": "mum"},
        ],
    )
    fp, out = _fixed(ped)
    assert fp.input_row.tolist() == [1, 2, 0]
    assert fp.ped_depth.tolist() == [0, 0, 1]
    assert out["id"].to_list() == ["1", "2", "3"]
    assert out["note"].to_list() == ["dad", "mum", "child"]


def test_sex_is_resolved_from_role(tmp_path):
    """A missing sex is imputed from parent role and recorded in sex_source."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "F", "mother": 2, "father": 1},
        ],
    )
    fp, out = _fixed(ped)
    assert fp.sex.tolist()[0] == SEX_MALE
    assert out["sex"].to_list() == ["1", "2", "2"]
    assert out["sex_source"].to_list()[0] == "imputed_from_missing"


def test_fill_half_founders_writes_phantoms_into_the_empty_slot(tmp_path):
    """A half-founder's missing parent slot names a new phantom founder; other slots keep their tokens."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 3, "sex": "F", "mother": -1, "father": 1},
        ],
    )
    fp, out = _fixed(ped, fill_half_founders=True)
    assert [(f["id"], f["sex"]) for f in fp.founders] == [(4, SEX_FEMALE)]
    assert fp.filled_mother.tolist() == [False, True]
    assert out["id"].to_list() == ["4", "1", "3"]
    assert out["mother"].to_list() == ["-1", "-1", "4"]
    assert out["father"].to_list() == ["-1", "-1", "1"]


def test_frame_matches_the_arrays(tmp_path):
    """frame() is the input rows in output order, with ID codes and int8 sex."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 3, "sex": "F", "mother": 2, "father": 1},
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
        ],
    )
    fp, _out = _fixed(ped)
    frame = fp.frame()
    assert frame.columns == ["id", "sex", "mother", "father", "sex_source", "ped_depth"]
    assert np.array_equal(frame["id"].to_numpy(), fp.ids)
    assert frame["sex"].to_list() == [SEX_MALE, SEX_FEMALE, SEX_FEMALE]


# ---------------------------------------------------------------------------
# fix_validated: validate's plain and --drop-offending paths without argv
# ---------------------------------------------------------------------------


def _outcome(path, **kwargs):
    _n, results, _findings, ctx = validate_pedigree(path)
    written: list = []

    def write(frame, fp):
        out = path.parent / "validate.tsv"
        frame.write_csv(out, separator="\t")
        written.append(frame)
        return out

    outcome = fix_validated(ctx, results, write=write, **kwargs)
    return outcome, written


_PLAIN = {"drop_offending": False, "fill_half_founders": False}
_DROP = {"drop_offending": True, "fill_half_founders": False}


def test_blocked_writes_nothing(tmp_path):
    """A BLOCK_OR_DROP FAIL blocks plain validate with its message; write is never called."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
        ],
    )
    outcome, written = _outcome(ped, **_PLAIN)
    assert outcome.exit_code == 2
    assert outcome.blocked == "fix the following before re-running:"
    assert outcome.block_lines == ["duplicate IDs detected"]
    assert written == []
    assert outcome.fixed is None


def test_plain_exit_code_follows_findings(tmp_path):
    """Exit 0 for a clean pedigree, 1 when a non-blocking Check FAILed (here, missing parents)."""
    clean = _write_ped(
        tmp_path / "clean.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
        ],
    )
    outcome, written = _outcome(clean, **_PLAIN)
    assert (outcome.exit_code, len(written)) == (0, 1)
    missing = _write_ped(tmp_path / "missing.tsv", [{"id": 3, "sex": "M", "mother": 1, "father": 2}])
    outcome, written = _outcome(missing, **_PLAIN)
    assert outcome.exit_code == 1
    assert written[0]["id"].to_list() == ["1", "2", "3"]


def test_drop_offending_reduces_and_self_verifies(tmp_path):
    """--drop-offending removes the flagged ids, writes the Reduced Pedigree, and reports the reduction."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "F", "mother": 2, "father": 3},  # self-loop
        ],
    )
    outcome, written = _outcome(ped, **_DROP)
    assert outcome.exit_code == 1
    assert outcome.reduction is not None
    # A self-loop is also a one-node cycle, so acyclic flags it too.
    assert {(fid, check) for fid, check, _rnd in outcome.reduction.dropped} == {("3", "self_loops"), ("3", "acyclic")}
    assert written[0]["id"].to_list() == ["1", "2"]


def test_drop_offending_with_nothing_to_drop_exits_zero(tmp_path):
    """A pedigree with only auto-fixable FAILs needs no reduction under --drop-offending."""
    ped = _write_ped(tmp_path / "p.tsv", [{"id": 3, "sex": "M", "mother": 1, "father": 2}])
    outcome, written = _outcome(ped, **_DROP)
    assert (outcome.exit_code, outcome.reduction, len(written)) == (0, None, 1)


def test_drop_offending_that_empties_the_pedigree_blocks(tmp_path):
    """Dropping every individual is BLOCKED rather than written as an empty file."""
    ped = _write_ped(tmp_path / "p.tsv", [{"id": 1, "sex": "M", "mother": 1, "father": -1}])
    outcome, written = _outcome(ped, **_DROP)
    assert outcome.exit_code == 2
    assert outcome.blocked == "--drop-offending removed every individual (empty pedigree)"
    assert written == []
