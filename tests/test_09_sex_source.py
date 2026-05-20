"""Tests for the 0.9 ``sex_source`` per-row column and override behaviour.

Verifies the four sex_source categories end-to-end:
- ``input`` — assertion preserved.
- ``imputed_from_missing`` — was unsexed; role implied F or M.
- ``imputed_from_role`` — asserted but topology disagreed; override fired.
- ``unresolved`` — still SEX_UNKNOWN after both imputation passes.
"""

from __future__ import annotations

import gzip

import pandas as pd
from conftest import run_pedsum
from conftest import write_ped as _write_ped


def _mixed_pedigree(path):
    """Pedigree exercising all four sex_source categories.

    id=1 input M; id=2 input F; id=3 input M (used as child).
    id=5 unsexed used only as mother of id=3 -> imputed_from_missing (F).
    id=6 asserted M used only as mother of id=7 -> imputed_from_role (F).
    id=7 input F (used as child).
    id=8 unsexed orphan -> unresolved (requires --allow-missing-sex).
    """
    return _write_ped(
        path,
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 5, "sex": "", "mother": -1, "father": -1},
            {"id": 6, "sex": "M", "mother": -1, "father": -1},
            {"id": 8, "sex": "", "mother": -1, "father": -1},
            {"id": 3, "sex": "M", "mother": 5, "father": 1},
            {"id": 7, "sex": "F", "mother": 6, "father": 1},
        ],
    )


def test_annotated_tsv_has_sex_source_column(tmp_path):
    """Summarize emits a per-row sex_source column in annotated.tsv.gz."""
    ped = _mixed_pedigree(tmp_path / "ped.tsv")
    out_dir = tmp_path / "out"
    r = run_pedsum(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--allow-missing-sex",
            "--no-inbreeding",
            "--no-effective-size",
        ]
    )
    assert r.returncode == 0, r.stderr
    with gzip.open(out_dir / "annotated.tsv.gz", "rt") as fh:
        ann = pd.read_csv(fh, sep="\t")
    assert "sex_source" in ann.columns
    by_id = ann.set_index("id")["sex_source"].to_dict()
    assert by_id[1] == "input"
    assert by_id[5] == "imputed_from_missing"
    assert by_id[6] == "imputed_from_role"
    assert by_id[8] == "unresolved"


def test_validate_tsv_has_sex_source_column(tmp_path):
    """Validate emits sex_source in validate.tsv.gz."""
    ped = _mixed_pedigree(tmp_path / "ped.tsv")
    out_dir = tmp_path / "out"
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--allow-missing-sex",
        ]
    )
    assert r.returncode == 0, r.stderr
    with gzip.open(out_dir / "validate.tsv.gz", "rt") as fh:
        fixed = pd.read_csv(fh, sep="\t", dtype=str)
    assert "sex_source" in fixed.columns
    by_id = fixed.set_index(fixed["id"].astype(int))["sex_source"].to_dict()
    assert by_id[1] == "input"
    assert by_id[5] == "imputed_from_missing"
    assert by_id[6] == "imputed_from_role"
    assert by_id[8] == "unresolved"
    # Row 6 in the fixed file has its sex normalised to F.
    row6 = fixed.loc[fixed["id"].astype(int) == 6].iloc[0]
    assert row6["sex"] == "F"


def test_no_override_asserted_sex_flag_blocks_contradictions_in_cli(tmp_path):
    """--no-override-asserted-sex restores 0.8's hard-block on sex/role contradictions."""
    ped = _write_ped(
        tmp_path / "ped.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},  # asserted M used as mother
            {"id": 3, "sex": "F", "mother": 2, "father": 1},
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
            "--no-override-asserted-sex",
            "--no-inbreeding",
            "--no-effective-size",
        ]
    )
    assert r.returncode == 1, r.stderr
    assert "sex_role_consistency" in r.stderr


def test_override_count_in_grouped_summary(tmp_path):
    """Validate's grouped stderr summary shows PASS (N overridden from role)."""
    ped = _write_ped(
        tmp_path / "ped.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},  # overridden to F
            {"id": 3, "sex": "F", "mother": 2, "father": 1},
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out_dir)])
    assert r.returncode == 0, r.stderr
    # One row overrides; expect "PASS (1 overridden from role)" on the
    # sex_role_consistency line in the grouped summary.
    lines = [line for line in r.stderr.splitlines() if "sex consistent with parent role" in line]
    assert lines, "expected sex_role_consistency line in grouped summary"
    assert "PASS" in lines[0]
    assert "1 overridden from role" in lines[0]
