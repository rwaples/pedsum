"""Tests for the silent topological reorder in validate output."""

from __future__ import annotations

import gzip

import pandas as pd
from conftest import load_validate_tsv_gz, run_pedsum
from conftest import write_ped as _write_ped

import pedigree_summary as ps


def test_topological_row_order_check_removed():
    """The legacy topological_row_order check no longer exists."""
    assert "topological_row_order" not in ps._CHECK_ORDER
    assert not hasattr(ps, "_check_topological_row_order")


def test_validate_reorders_unordered_tsv(tmp_path):
    """Child-before-parent input → fixed output is parents-first; INFO log fires."""
    # Child (id=3) appears before its parents (1, 2). We expect the fixed
    # output to put the founders first.
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 3, "sex": "M", "mother": 2, "father": 1},
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 0, r.stderr
    assert "reordering" in r.stderr.lower()
    with gzip.open(tmp_path / "out" / "validate.tsv.gz", "rt") as fh:
        fixed = pd.read_csv(fh, sep="\t", dtype=str)
    # The two founders must appear before the child in row order.
    ids_in_order = fixed["id"].astype(int).tolist()
    assert ids_in_order.index(1) < ids_in_order.index(3)
    assert ids_in_order.index(2) < ids_in_order.index(3)


def test_validate_no_reorder_when_already_ordered(tmp_path):
    """In-order input produces no 'reordering' INFO log line."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "M", "mother": 2, "father": 1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 0, r.stderr
    assert "reordering" not in r.stderr.lower()


def test_validate_does_not_reorder_on_cycle(tmp_path):
    """Cyclic input surfaces acyclic FAIL with no traceback and no fixed TSV."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            # Cycle: id 1 has mother 2, id 2 has mother 1.
            {"id": 1, "sex": "F", "mother": 2, "father": -1},
            {"id": 2, "sex": "F", "mother": 1, "father": -1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 2, r.stderr
    assert "Traceback" not in r.stderr
    assert "cycle" in r.stderr.lower() or "acyclic" in r.stderr.lower()


def test_validate_fixed_tsv_sex_source_rides_with_reorder(tmp_path):
    """sex_source column stays aligned with its id after topological reorder.

    Input is child-before-parent; one founder (id=5) has empty sex and is
    used as a mother, so the 0.8 missing→F imputation fires. After the
    topological reorder, the row carrying sex_source="imputed_from_missing"
    must still be the id=5 row with sex normalised to "F".
    """
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 3, "sex": "F", "mother": 5, "father": 1},  # child first
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 5, "sex": "", "mother": -1, "father": -1},  # unsexed mother
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out_dir)])
    assert r.returncode == 0, r.stderr
    fixed = load_validate_tsv_gz(out_dir)
    assert "sex_source" in fixed.columns
    by_id = fixed.set_index(fixed["id"].astype(int))
    assert by_id.loc[5, "sex"] == "F"
    assert by_id.loc[5, "sex_source"] == "imputed_from_missing"
    # The other rows keep their asserted sex.
    assert by_id.loc[1, "sex"] == "M"
    assert by_id.loc[1, "sex_source"] == "input"
    # Reorder happened: founders precede the child.
    ids_in_order = fixed["id"].astype(int).tolist()
    assert ids_in_order.index(1) < ids_in_order.index(3)
    assert ids_in_order.index(5) < ids_in_order.index(3)


def test_validate_fixed_tsv_overridden_sex_rides_with_reorder(tmp_path):
    """sex_source="imputed_from_role" also rides through the reorder.

    Same shape as above but id=5 is asserted M and used only as a mother,
    triggering the 0.9 override-asserted-sex path. Output row must show
    sex="F" and sex_source="imputed_from_role" after the reorder.
    """
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 3, "sex": "F", "mother": 5, "father": 1},
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 5, "sex": "M", "mother": -1, "father": -1},  # asserted M as mother
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out_dir)])
    assert r.returncode == 0, r.stderr
    fixed = load_validate_tsv_gz(out_dir)
    by_id = fixed.set_index(fixed["id"].astype(int))
    assert by_id.loc[5, "sex"] == "F"
    assert by_id.loc[5, "sex_source"] == "imputed_from_role"
    ids_in_order = fixed["id"].astype(int).tolist()
    assert ids_in_order.index(5) < ids_in_order.index(3)
