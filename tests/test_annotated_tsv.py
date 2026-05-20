"""Tests for ``annotated.tsv.gz`` topological realignment and column collisions.

Pins two behaviours of ``_write_annotated_tsv``:
- when the input pedigree is not in topological order, rows are realigned
  to match the topologically-ordered ``idf`` table and arbitrary input
  columns ride along by id;
- when an input column collides with a pedsum-derived column, the input
  is preserved under ``<name>_input`` and a warning is logged.
"""

from __future__ import annotations

from conftest import load_annotated_tsv_gz, run_pedsum
from conftest import write_ped as _write_ped


def test_annotated_tsv_realigns_to_topological_order(tmp_path):
    """Child-before-parent input → output is topological; extras ride along."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "tag": "child"},
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "tag": "dad"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "tag": "mom"},
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--no-inbreeding",
            "--no-effective-size",
        ]
    )
    assert r.returncode == 0, r.stderr
    ann = load_annotated_tsv_gz(out_dir)
    ids = ann["id"].tolist()
    # Every parent id must appear at a strictly smaller index than the child.
    assert ids.index(1) < ids.index(3)
    assert ids.index(2) < ids.index(3)
    # Extra column rides along by id, not by original row position.
    by_id = dict(zip(ann["id"].tolist(), ann["tag"].tolist(), strict=True))
    assert by_id == {1: "dad", 2: "mom", 3: "child"}


def test_annotated_tsv_renames_colliding_input_column(tmp_path):
    """Input ``n_descendants`` column → kept as ``n_descendants_input``."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "n_descendants": "alpha"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "n_descendants": "beta"},
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "n_descendants": "gamma"},
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--no-inbreeding",
            "--no-effective-size",
        ]
    )
    assert r.returncode == 0, r.stderr
    ann = load_annotated_tsv_gz(out_dir)
    # Both columns present: the derived integer column and the renamed input.
    assert "n_descendants" in ann.columns
    assert "n_descendants_input" in ann.columns
    by_id = dict(
        zip(ann["id"].tolist(), ann["n_descendants_input"].tolist(), strict=True),
    )
    assert by_id == {1: "alpha", 2: "beta", 3: "gamma"}
    # Derived column carries actual descendant counts (founders have 1, child has 0).
    derived = dict(zip(ann["id"].tolist(), ann["n_descendants"].tolist(), strict=True))
    assert derived[3] == 0  # child has no descendants
    assert derived[1] >= 1  # both founders have at least one descendant
    assert derived[2] >= 1
    # Warning logged for the rename.
    assert "n_descendants" in r.stderr
    assert "n_descendants_input" in r.stderr


def test_annotated_tsv_renames_component_id_collision(tmp_path):
    """``component_id`` collision is handled the same way (more than one rename)."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1, "component_id": "x"},
            {"id": 2, "sex": "F", "mother": -1, "father": -1, "component_id": "y"},
            {"id": 3, "sex": "M", "mother": 2, "father": 1, "component_id": "z"},
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--no-inbreeding",
            "--no-effective-size",
        ]
    )
    assert r.returncode == 0, r.stderr
    ann = load_annotated_tsv_gz(out_dir)
    assert "component_id" in ann.columns
    assert "component_id_input" in ann.columns
    by_id = dict(
        zip(ann["id"].tolist(), ann["component_id_input"].tolist(), strict=True),
    )
    assert by_id == {1: "x", 2: "y", 3: "z"}
