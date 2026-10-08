"""Tests for the 0.9 ``sex_source`` per-row column and override behaviour.

Verifies the four sex_source categories end-to-end:
- ``input`` — assertion preserved.
- ``imputed_from_missing`` — was unsexed; role implied F or M.
- ``imputed_from_role`` — asserted but topology disagreed; override fired.
- ``unresolved`` — still SEX_UNKNOWN after both imputation passes.
"""

from __future__ import annotations

import gzip

import polars as pl
import pytest
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
        ]
    )
    assert r.returncode == 0, r.stderr
    with gzip.open(out_dir / "annotated.tsv.gz", "rb") as fh:
        ann = pl.read_csv(fh.read(), separator="\t")
    assert "sex_source" in ann.columns
    by_id = dict(zip(ann["id"].to_list(), ann["sex_source"].to_list(), strict=True))
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
    with gzip.open(out_dir / "validate.tsv.gz", "rb") as fh:
        fixed = pl.read_csv(fh.read(), separator="\t", infer_schema=False)
    assert "sex_source" in fixed.columns
    by_id = dict(zip(fixed["id"].cast(pl.Int64).to_list(), fixed["sex_source"].to_list(), strict=True))
    assert by_id[1] == "input"
    assert by_id[5] == "imputed_from_missing"
    assert by_id[6] == "imputed_from_role"
    assert by_id[8] == "unresolved"
    # Row 6 in the fixed file has its sex overridden to female (PLINK 2).
    row6 = fixed.filter(pl.col("id").cast(pl.Int64) == 6).row(0, named=True)
    assert row6["sex"] == "2"


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


# Sex tokens per input style: (female, male, unknown).
_SEX_TOKENS = {
    "words": ("F", "M", ""),
    "plink": ("2", "1", "0"),
    "plink_minus_one": ("2", "1", "-1"),
}


@pytest.mark.parametrize("encoding", sorted(_SEX_TOKENS))
def test_validate_tsv_writes_one_sex_encoding(tmp_path, encoding):
    """validate.tsv.gz writes every row's sex in PLINK coding: 1=male, 2=female, 0=unknown.

    Rows kept from the input, imputed rows, unresolved rows and added founders
    share the one encoding, whatever the input used, so a reader that infers
    types sees an integer column (issues #11, #12).
    """
    f, m, u = _SEX_TOKENS[encoding]
    ped = _write_ped(
        tmp_path / "ped.tsv",
        [
            {"id": 1, "sex": m, "mother": -1, "father": -1},
            {"id": 2, "sex": f, "mother": -1, "father": -1},
            {"id": 5, "sex": u, "mother": -1, "father": -1},  # mother -> imputed female
            {"id": 6, "sex": u, "mother": -1, "father": -1},  # father -> imputed male
            {"id": 8, "sex": u, "mother": -1, "father": -1},  # orphan -> unresolved
            {"id": 3, "sex": m, "mother": 5, "father": 6},
            {"id": 7, "sex": f, "mother": 2, "father": 1},
            {"id": 9, "sex": f, "mother": 100, "father": 1},  # 100 -> added female founder
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out_dir), "--allow-missing-sex"])
    assert r.returncode == 1, r.stderr  # the missing parent is a finding
    with gzip.open(out_dir / "validate.tsv.gz", "rb") as fh:
        fixed = pl.read_csv(fh.read(), separator="\t")
    assert fixed["sex"].dtype.is_integer()
    by_id = dict(zip(fixed["id"].to_list(), fixed["sex"].to_list(), strict=True))
    assert by_id == {1: 1, 2: 2, 5: 2, 6: 1, 8: 0, 3: 1, 7: 2, 9: 2, 100: 2}


def test_drop_offending_self_verifies_plink_input(tmp_path):
    """--drop-offending re-reads its PLINK-coded output and self-verifies on a PLINK input."""
    ped = _write_ped(
        tmp_path / "ped.tsv",
        [
            {"id": 1, "sex": "2", "mother": -1, "father": -1},
            {"id": 2, "sex": "1", "mother": -1, "father": -1},
            {"id": 3, "sex": "2", "mother": -1, "father": -1},  # mother and father -> dropped
            {"id": 4, "sex": "2", "mother": 3, "father": 2},
            {"id": 5, "sex": "1", "mother": 1, "father": 3},
        ],
    )
    out_dir = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out_dir), "--drop-offending"])
    assert r.returncode == 1, r.stderr  # exit 1 because something was dropped
    assert "self-verify failed" not in r.stderr
    with gzip.open(out_dir / "validate.tsv.gz", "rb") as fh:
        fixed = pl.read_csv(fh.read(), separator="\t")
    assert dict(zip(fixed["id"].to_list(), fixed["sex"].to_list(), strict=True)) == {1: 2, 2: 1, 4: 2, 5: 1}
