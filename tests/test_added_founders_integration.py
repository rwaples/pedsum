"""End-to-end CLI tests that synthesised founders appear in validate.tsv.gz."""

from __future__ import annotations

from conftest import load_validate_tsv_gz, run_pedsum
from conftest import write_ped as _write_ped


def test_validate_writes_added_mother_founder(tmp_path):
    """Mother referenced but absent → row appears in fixed TSV with sex=F."""
    # id=100 is referenced as mother of id=3 but never appears as an id.
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 3, "sex": "F", "mother": 100, "father": 1},
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out_dir)])
    # Missing parent triggers a finding (rc=1) but does not block the
    # auto-fix; validate.tsv.gz is written with the synthesised founder.
    assert r.returncode == 1, r.stderr
    assert "mother IDs present in pedigree" in r.stderr
    fixed = load_validate_tsv_gz(out_dir)
    by_id = {int(row["id"]): row for _, row in fixed.iterrows()}
    assert 100 in by_id
    added = by_id[100]
    assert added["sex"] == "F"
    assert added["mother"] == "-1"
    assert added["father"] == "-1"


def test_validate_writes_added_father_founder(tmp_path):
    """Father referenced but absent → row appears in fixed TSV with sex=M."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "F", "mother": 2, "father": 100},
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out_dir)])
    assert r.returncode == 1, r.stderr
    assert "father IDs present in pedigree" in r.stderr
    fixed = load_validate_tsv_gz(out_dir)
    by_id = {int(row["id"]): row for _, row in fixed.iterrows()}
    assert 100 in by_id
    assert by_id[100]["sex"] == "M"


def test_validate_writes_added_conflicting_founder_with_no_sex_check(tmp_path):
    """Both-roles missing id is synthesised as F when --no-sex-check is set.

    Without --no-sex-check this case BLOCKs on parent_refs_sex_conflict
    (rc=2 and no fixed TSV). With the flag, the conflict check is
    bypassed and ``_build_added_founders`` synthesises a sex=F founder
    per its documented fallback.
    """
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "M", "mother": 100, "father": 1},  # 100 as mother
            {"id": 4, "sex": "F", "mother": 2, "father": 100},  # 100 as father
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--no-sex-check",
        ]
    )
    assert r.returncode == 1, r.stderr
    # The conflict check renders as SKIP (bypassed via --no-sex-check).
    assert "bypassed via --no-sex-check" in r.stderr
    fixed = load_validate_tsv_gz(out_dir)
    by_id = {int(row["id"]): row for _, row in fixed.iterrows()}
    assert 100 in by_id
    assert by_id[100]["sex"] == "F"
