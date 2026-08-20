"""Behavior tests for ``validate --drop-offending`` (Reduced Pedigree).

The flag removes the individuals named in droppable check Findings (clearing
references to them), iterating to a fixpoint, and emits a pedigree that passes
under the invoked flags. See docs/adr/0003-drop-offending-reduction.md.
"""

from __future__ import annotations

import polars as pl
from conftest import load_validate_tsv_gz, run_pedsum
from conftest import write_ped as _write_ped


def _read_manifest(out_dir) -> pl.DataFrame:
    """Read ``validate.dropped.tsv`` from an output dir."""
    return pl.read_csv(out_dir / "validate.dropped.tsv", separator="\t")


def _role_conflict_pedigree(path):
    """id=3 (F) used as a mother (id 4) and a father (ids 5, 6); has an extra column."""
    return _write_ped(
        path,
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1, "note": "a"},
            {"id": 2, "sex": "M", "mother": -1, "father": -1, "note": "b"},
            {"id": 3, "sex": "F", "mother": -1, "father": -1, "note": "c"},
            {"id": 4, "sex": "F", "mother": 3, "father": 2, "note": "d"},
            {"id": 5, "sex": "M", "mother": 1, "father": 3, "note": "e"},
            {"id": 6, "sex": "M", "mother": 1, "father": 3, "note": "f"},
        ],
    )


def test_drop_role_conflict_reduces_and_passes(tmp_path):
    """A sex_role_consistency offender is dropped and the reduced pedigree passes."""
    ped = _role_conflict_pedigree(tmp_path / "p.tsv")
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    assert r.returncode == 1, r.stderr  # exit 1 because something was dropped
    manifest = _read_manifest(out)
    assert manifest["id"].to_list() == [3]
    assert manifest["check"].to_list() == ["sex_role_consistency"]
    assert manifest["round"].to_list() == [1]
    r2 = run_pedsum(["validate", "--in", str(out / "validate.tsv.gz"), "--out", str(tmp_path / "out2")])
    assert r2.returncode == 0, r2.stderr  # reduced pedigree re-validates clean


def test_drop_preserves_extra_columns_and_clears_refs(tmp_path):
    """Dropping keeps unrelated columns and clears (not resurrects) refs to the offender."""
    ped = _role_conflict_pedigree(tmp_path / "p.tsv")
    out = tmp_path / "out"
    run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    df = load_validate_tsv_gz(out)
    assert "3" not in df["id"].to_list()  # offender gone
    assert "note" in df.columns  # extra column survives
    assert df.filter(pl.col("id") == "4")["mother"][0] == "-1"  # ref to 3 cleared
    assert df.filter(pl.col("id") == "5")["father"][0] == "-1"


def test_drop_spawned_offender_reaches_fixpoint(tmp_path):
    """Dropping a child orphans its unsexed parent → new unknown_sex in round 2."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 8, "sex": "M", "mother": -1, "father": -1},
            {"id": 9, "sex": "", "mother": -1, "father": -1},
            {"id": 10, "sex": "F", "mother": 9, "father": 8},
            {"id": 11, "sex": "F", "mother": 10, "father": 8},
            {"id": 12, "sex": "M", "mother": 8, "father": 10},
        ],
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    assert r.returncode == 1, r.stderr
    manifest = _read_manifest(out)
    rounds = dict(zip(manifest["id"], manifest["round"], strict=False))
    assert rounds[9] == 2  # 9 only droppable once its child 10 is removed
    assert max(manifest["round"]) == 2
    assert "2 round(s)" in r.stderr
    # An INFO line announces the loop start before the per-round detail.
    assert "starting reduction loop" in r.stderr
    # Per-round INFO lines list the changes made each iteration.
    assert "drop-offending round 1:" in r.stderr
    assert "drop-offending round 2:" in r.stderr
    assert "unknown_sex" in r.stderr


def test_drop_acyclic_drops_cycle_members_only(tmp_path):
    """A 2-cycle (1<->2) drops only the cycle members; the descendant survives (SCC-only)."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": 2, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": 1},
            {"id": 3, "sex": "F", "mother": 1, "father": -1},
        ],
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending", "--allow-missing-sex"])
    assert r.returncode == 1, r.stderr
    df = load_validate_tsv_gz(out)
    assert df["id"].to_list() == ["3"]  # descendant survives
    assert df.filter(pl.col("id") == "3")["mother"][0] == "-1"  # ref to dropped 1 cleared
    assert set(_read_manifest(out)["id"]) == {1, 2}  # only cycle members dropped


def test_drop_duplicate_drops_all_copies(tmp_path):
    """duplicate_ids removes every row carrying the duplicated id."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},
            {"id": 5, "sex": "F", "mother": 2, "father": 1},
            {"id": 5, "sex": "M", "mother": -1, "father": -1},
        ],
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    assert r.returncode == 1, r.stderr
    assert "5" not in load_validate_tsv_gz(out)["id"].to_list()  # both copies removed
    assert _read_manifest(out)["id"].to_list() == [5]  # one distinct id


def test_drop_composes_with_allow_missing_sex(tmp_path):
    """--allow-missing-sex keeps an unsexed orphan (-1); drop removes only what still fails."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},
            {"id": 3, "sex": "", "mother": -1, "father": -1},  # orphan unsexed: tolerated, kept as -1
            {"id": 5, "sex": "F", "mother": -1, "father": -1},  # both-role offender
            {"id": 6, "sex": "F", "mother": 5, "father": 2},  # 5 used as mother
            {"id": 7, "sex": "M", "mother": 1, "father": 5},  # ...and father
        ],
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending", "--allow-missing-sex"])
    assert r.returncode == 1, r.stderr
    dropped = set(_read_manifest(out)["id"])
    assert 5 in dropped  # offender dropped
    assert 3 not in dropped  # tolerated orphan kept
    # The reduced output keeps id 3 as -1; that it re-validated clean during
    # self-verify proves the verify ran under --allow-missing-sex (plain
    # validate would FAIL on the unresolved-sex row).
    assert load_validate_tsv_gz(out).filter(pl.col("id") == "3")["sex"][0] == "-1"


def test_drop_no_sex_check_does_not_drop_missing_parent_conflict(tmp_path):
    """--no-sex-check tolerates parent_refs_sex_conflict, so the parent is synthesized."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},
            {"id": 4, "sex": "F", "mother": 99, "father": 2},  # 99 missing, used as mother
            {"id": 5, "sex": "M", "mother": 1, "father": 99},  # ...and father
        ],
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending", "--no-sex-check"])
    assert r.returncode == 0, r.stderr  # nothing dropped; conflict tolerated
    assert _read_manifest(out).is_empty()
    assert "99" in load_validate_tsv_gz(out)["id"].to_list()  # synthesized founder, refs not cleared


def test_drop_non_reducible_failure_blocks(tmp_path):
    """A column/parse-level failure (negative id) still BLOCKs under --drop-offending."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
            {"id": -5, "sex": "M", "mother": -1, "father": -1},
            {"id": 3, "sex": "F", "mother": 3, "father": 1},  # also a self-loop (droppable)
        ],
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    assert r.returncode == 2, r.stderr
    assert "BLOCKED" in r.stderr
    assert "negative_ids" in r.stderr
    assert not (out / "validate.tsv.gz").exists()


def test_drop_all_rows_blocks_empty(tmp_path):
    """Reduction that removes every individual fails cleanly (empty pedigree)."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": 1, "father": -1},  # self-loop
            {"id": 2, "sex": "M", "mother": 2, "father": -1},  # self-loop
        ],
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    assert r.returncode == 2, r.stderr
    assert "empty pedigree" in r.stderr.lower()
    assert not (out / "validate.tsv.gz").exists()


def test_drop_clean_input_writes_header_only_manifest(tmp_path):
    """A clean input drops nothing: exit 0 and a header-only manifest."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "F", "mother": 2, "father": 1},
        ],
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    assert r.returncode == 0, r.stderr
    manifest = _read_manifest(out)
    assert manifest.is_empty()
    assert list(manifest.columns) == ["id", "check", "round"]
    assert (out / "validate.tsv.gz").exists()


def test_drop_large_fraction_warns(tmp_path):
    """Dropping >10% of rows emits a warning."""
    ped = _role_conflict_pedigree(tmp_path / "p.tsv")  # drops 1 of 6 = 16.7%
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    assert "removed 16.7% of rows" in r.stderr


def test_drop_off_by_default_unchanged(tmp_path):
    """Without the flag, a non-blocking sex_role_consistency FAIL is unchanged (exit 1, kept)."""
    ped = _role_conflict_pedigree(tmp_path / "p.tsv")
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out)])
    assert r.returncode == 1, r.stderr
    assert not (out / "validate.dropped.tsv").exists()
    assert "3" in load_validate_tsv_gz(out)["id"].to_list()  # offender NOT dropped
