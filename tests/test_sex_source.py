"""The per-row ``sex_source`` column and the asserted-sex override, end to end.

The four sex_source categories:
- ``input`` — assertion preserved.
- ``imputed_from_missing`` — was unsexed; role implied F or M.
- ``imputed_from_role`` — asserted but topology disagreed; override fired.
- ``unresolved`` — still SEX_UNKNOWN after both imputation passes.
"""

from __future__ import annotations

import polars as pl
import pytest
from conftest import load_annotated_tsv_gz, load_summary_yaml, load_validate_tsv_gz, read_tsv_gz, run_pedsum
from conftest import write_ped as _write_ped


def _all_sex_sources_pedigree(path):
    """Pedigree exercising all four sex_source categories.

    id=1 input M; id=2 input F; id=3 input M (used as child).
    id=5 unsexed used only as mother of id=3 -> imputed_from_missing (F).
    id=6 asserted M used only as mother of id=7 -> imputed_from_role (F).
    id=7 input F (used as child).
    id=8 unsexed, never a parent -> unresolved (kept with sex 0).
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
    """Summarizing validate.tsv.gz keeps its sex_source in annotated.tsv.gz and counts it in summary.yaml.

    Loading the Fixed Pedigree again finds every sex resolved, so the
    provenance comes from the input's column (ADR 0006).
    """
    ped = _all_sex_sources_pedigree(tmp_path / "ped.tsv")
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "v")])
    assert r.returncode == 1, r.stderr  # the override is a repair
    out_dir = tmp_path / "out"
    fixed = tmp_path / "v" / "validate.tsv.gz"
    r = run_pedsum(["summarize", "--in", str(fixed), "--out", str(out_dir), "--no-inbreeding"])
    assert r.returncode == 0, r.stderr
    size = load_summary_yaml(out_dir)["pedigree"]["structure"]["size_structure"]
    assert size["sex_source"] == {"input": 4, "imputed_from_missing": 1, "imputed_from_role": 1, "unresolved": 1}
    ann = load_annotated_tsv_gz(out_dir)
    assert "sex_source" in ann.columns
    by_id = dict(zip(ann["id"].to_list(), ann["sex_source"].to_list(), strict=True))
    assert by_id[1] == "input"
    assert by_id[5] == "imputed_from_missing"
    assert by_id[6] == "imputed_from_role"
    assert by_id[8] == "unresolved"


def test_validate_tsv_has_sex_source_column(tmp_path):
    """Validate emits sex_source in validate.tsv.gz."""
    ped = _all_sex_sources_pedigree(tmp_path / "ped.tsv")
    out_dir = tmp_path / "out"
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
        ]
    )
    assert r.returncode == 1, r.stderr  # the override is a sex_role_override Finding
    fixed = load_validate_tsv_gz(out_dir)
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
    """Under --no-override-asserted-sex a contradiction is a sex_role_consistency FAIL, so summarize refuses."""
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


def test_override_is_a_finding(tmp_path):
    """Each override is a sex_role_override Finding: validate.log has it and validate exits 1 (ADR 0006)."""
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
    assert r.returncode == 1, r.stderr
    lines = [line for line in r.stderr.splitlines() if "no recorded sex overridden by role" in line]
    assert lines[0].endswith("FAIL (1)")
    consistency = [line for line in r.stderr.splitlines() if "sex consistent with parent role" in line]
    assert consistency[0].endswith("PASS")
    log = pl.read_csv(out_dir / "validate.log", separator="\t", infer_schema=False)
    assert log.rows() == [
        ("sex_role_override", "2", "1", "id=2 recorded M but used only as mother in row(s) [2]; sex set to F", None)
    ]
    # --drop-offending keeps the overridden individual: an autofix-only run exits 0.
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "d"), "--drop-offending"])
    assert r.returncode == 0, r.stderr
    assert load_validate_tsv_gz(tmp_path / "d")["id"].to_list() == ["1", "2", "3"]
    # --no-override-asserted-sex SKIPs the Check.
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "n"), "--no-override-asserted-sex"])
    line = next(line for line in r.stderr.splitlines() if "no recorded sex overridden by role" in line)
    assert line.endswith("SKIP (off (--no-override-asserted-sex))")


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
    fixed = read_tsv_gz(out_dir / "validate.tsv.gz")
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
    fixed = read_tsv_gz(out_dir / "validate.tsv.gz")
    assert dict(zip(fixed["id"].to_list(), fixed["sex"].to_list(), strict=True)) == {1: 2, 2: 1, 4: 2, 5: 1}


def test_safe_attempt_nulls_small_sex_source_counts():
    """--safe-attempt nulls a sex_source count below the minimum cell, like the other size counts."""
    from pedsum.sections import redact_size_structure

    sizes = {
        "next_components": [],
        "depth_counts": [],
        "largest_component": 100,
        "sex_source": {"input": 90, "imputed_from_missing": 0, "imputed_from_role": 3, "unresolved": 7},
    }
    redact_size_structure(sizes, 5, 100)
    assert sizes["sex_source"] == {"input": 90, "imputed_from_missing": 0, "imputed_from_role": None, "unresolved": 7}
