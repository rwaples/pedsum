"""``validate --fill-half-founders``: one phantom founder per missing parent slot (pedsum issue #9)."""

from __future__ import annotations

import numpy as np
import yaml
from conftest import load_validate_tsv_gz, run_pedsum, write_ped
from pedigree_graph import PedigreeGraph

from pedsum.base import SEX_FEMALE, SEX_MALE
from pedsum.report import _build_phantom_parents, _next_free_id

# 1-4 founders; 5 and 6 are full sibs; 7 (a daughter of 5) lacks a father and
# 8 (a son of 6) lacks a mother, so their child 9 is inbred; 10 lacks a father
# and has a mother (100) that is referenced but absent; 11 is a child of sibs
# 5 x 6.
ROWS = [
    {"id": 1, "sex": "F", "mother": -1, "father": -1},
    {"id": 2, "sex": "M", "mother": -1, "father": -1},
    {"id": 3, "sex": "F", "mother": -1, "father": -1},
    {"id": 4, "sex": "M", "mother": -1, "father": -1},
    {"id": 5, "sex": "F", "mother": 1, "father": 2},
    {"id": 6, "sex": "M", "mother": 1, "father": 2},
    {"id": 7, "sex": "F", "mother": 5, "father": -1},
    {"id": 8, "sex": "M", "mother": -1, "father": 6},
    {"id": 9, "sex": "F", "mother": 7, "father": 8},
    {"id": 10, "sex": "M", "mother": 100, "father": -1},
    {"id": 11, "sex": "F", "mother": 5, "father": 6},
]


def _validate(tmp_path, *extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ped = write_ped(tmp_path / "p.tsv", ROWS)
    out_dir = tmp_path / "out"
    res = run_pedsum(["validate", "--in", str(ped), "--out", str(out_dir), *extra])
    return res, out_dir


def _parents(df):
    ids = df["id"].cast(int).to_numpy()
    mothers = df["mother"].cast(int).to_numpy()
    fathers = df["father"].cast(int).to_numpy()
    return ids, mothers, fathers


def test_phantoms_fill_each_missing_slot_once():
    """Each half-founder gets its own phantom, sexed by the slot, above every used ID."""
    ids = np.array([1, 2, 3, 4, 5])
    mothers = np.array([-1, -1, 1, -1, 50])
    fathers = np.array([-1, -1, -1, 2, -1])
    assert _next_free_id(ids, mothers, fathers) == 51
    new_m, new_f, phantoms = _build_phantom_parents(mothers, fathers, 51)
    assert phantoms == [{"id": 51, "sex": SEX_MALE}, {"id": 52, "sex": SEX_FEMALE}, {"id": 53, "sex": SEX_MALE}]
    np.testing.assert_array_equal(new_m, [-1, -1, 1, 52, 50])
    np.testing.assert_array_equal(new_f, [-1, -1, 51, 2, 53])
    # The inputs are not mutated.
    np.testing.assert_array_equal(mothers, [-1, -1, 1, -1, 50])
    np.testing.assert_array_equal(fathers, [-1, -1, -1, 2, -1])


def test_no_half_founders_adds_nothing():
    """A pedigree with only full founders and complete parentage is returned as is."""
    mothers = np.array([-1, -1, 1])
    fathers = np.array([-1, -1, 2])
    new_m, new_f, phantoms = _build_phantom_parents(mothers, fathers, 4)
    assert phantoms == []
    np.testing.assert_array_equal(new_m, mothers)
    np.testing.assert_array_equal(new_f, fathers)


def test_default_leaves_half_founders(tmp_path):
    """Without the flag, validate writes the half-founders unchanged."""
    res, out_dir = _validate(tmp_path)
    assert res.returncode == 1, res.stderr  # the absent mother 100 is a finding
    _ids, mothers, fathers = _parents(load_validate_tsv_gz(out_dir))
    assert int(((mothers == -1) ^ (fathers == -1)).sum()) == 3


def test_fill_leaves_only_full_founders(tmp_path):
    """With the flag, every half-founder points at a new founder and the output re-validates clean."""
    res, out_dir = _validate(tmp_path, "--fill-half-founders")
    assert res.returncode == 1, res.stderr
    fixed = load_validate_tsv_gz(out_dir)
    ids, mothers, fathers = _parents(fixed)
    assert not ((mothers == -1) ^ (fathers == -1)).any()
    # 11 input rows, the absent mother 100, and phantoms 101-103 in row order.
    assert len(fixed) == 15
    by_id = {int(r["id"]): r for r in fixed.iter_rows(named=True)}
    assert by_id[7]["father"] == "101"
    assert by_id[8]["mother"] == "102"
    assert by_id[10]["father"] == "103"
    assert [by_id[p]["sex"] for p in (101, 102, 103)] == ["1", "0", "1"]
    for p in (100, 101, 102, 103):
        assert (by_id[p]["mother"], by_id[p]["father"]) == ("-1", "-1")
    # Parents precede children, so the file feeds back into validate cleanly.
    row = {int(i): k for k, i in enumerate(ids)}
    assert all(row[p] < row[c] for c, p in zip(ids, mothers, strict=True) if p != -1)
    assert all(row[p] < row[c] for c, p in zip(ids, fathers, strict=True) if p != -1)
    again = run_pedsum(["validate", "--in", str(out_dir / "validate.tsv.gz"), "--out", str(tmp_path / "again")])
    assert again.returncode == 0, again.stderr


def _kinship_by_id(df, keep):
    ids, mothers, fathers = _parents(df)
    pg = PedigreeGraph.from_arrays(ids=ids, mother_ids=mothers, father_ids=fathers)
    k = pg.kinship_matrix().toarray()
    rows = [int(np.flatnonzero(ids == i)[0]) for i in keep]
    return k[np.ix_(rows, rows)]


def test_fill_preserves_kinship_among_input_individuals(tmp_path):
    """Phantoms are unrelated founders, so kinship (and F) among input rows is unchanged."""
    _res, plain = _validate(tmp_path / "plain")
    _res, filled = _validate(tmp_path / "filled", "--fill-half-founders")
    keep = [r["id"] for r in ROWS] + [100]
    before = _kinship_by_id(load_validate_tsv_gz(plain), keep)
    after = _kinship_by_id(load_validate_tsv_gz(filled), keep)
    np.testing.assert_array_equal(before, after)
    # The pedigree is not trivially outbred: 11 is a full-sib mating.
    assert before[keep.index(11), keep.index(11)] > 0.5


def _long_term(tmp_path, ped):
    out_dir = tmp_path / "es"
    res = run_pedsum(
        ["effective-size", "--in", str(ped), "--out", str(out_dir), "--estimators", "ne_long_term_contributions"]
    )
    assert res.returncode == 0, res.stderr
    return yaml.safe_load((out_dir / "effective_size.yaml").read_text())["effective_size"]["ne_long_term_contributions"]


def test_fill_unblocks_long_term_contributions(tmp_path):
    """Issue #9: the estimator refuses half-founders and runs once they are filled."""
    _res, plain = _validate(tmp_path / "plain")
    refused = _long_term(tmp_path / "plain", plain / "validate.tsv.gz")
    assert refused["ne"] is None
    assert refused["code"] == "incomplete_parentage"

    _res, filled = _validate(tmp_path / "filled", "--fill-half-founders")
    record = _long_term(tmp_path / "filled", filled / "validate.tsv.gz")
    assert record["ne"] is not None, record


def test_fill_applies_to_reduced_pedigree(tmp_path):
    """--drop-offending turns the descendant of a dropped cycle into a half-founder; the fill covers it.

    The phantom is numbered above the dropped IDs too, so it never reuses one
    that validate.dropped.tsv names.
    """
    ped = write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 8, "sex": "F", "mother": 9, "father": -1},
            {"id": 9, "sex": "M", "mother": -1, "father": 8},
            {"id": 3, "sex": "F", "mother": 8, "father": 4},
            {"id": 4, "sex": "M", "mother": -1, "father": -1},
        ],
    )
    out = tmp_path / "out"
    args = ["validate", "--in", str(ped), "--out", str(out), "--drop-offending", "--allow-missing-sex"]
    res = run_pedsum([*args, "--fill-half-founders"])
    assert res.returncode == 1, res.stderr  # something was dropped; self-verify passed
    fixed = load_validate_tsv_gz(out)
    assert sorted(fixed["id"].cast(int).to_list()) == [3, 4, 10]
    by_id = {int(r["id"]): r for r in fixed.iter_rows(named=True)}
    assert (by_id[3]["mother"], by_id[3]["father"]) == ("10", "4")
    assert by_id[10]["sex"] == "0"
