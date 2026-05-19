"""Unit + behavior tests for sex auto-detection and missing-sex imputation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pedigree_summary as ps
import pytest
from conftest import run_pedsum
from conftest import write_ped as _write_ped

# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_default_encoding(caplog):
    """0/1 sex column resolves to default encoding, no WARNING."""
    s = pd.Series(["0", "1", "0", "1"])
    with caplog.at_level(logging.INFO, logger="pedigree_summary"):
        out = ps._decode_sex(s, encoding="auto")
    assert out.tolist() == [ps.SEX_FEMALE, ps.SEX_MALE, ps.SEX_FEMALE, ps.SEX_MALE]
    assert any("default" in r.message for r in caplog.records if r.levelno == logging.INFO)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_auto_detect_plink_encoding(caplog):
    """1/2 sex column resolves to plink encoding, no WARNING."""
    s = pd.Series(["1", "2", "1", "2"])
    with caplog.at_level(logging.INFO, logger="pedigree_summary"):
        out = ps._decode_sex(s, encoding="auto")
    assert out.tolist() == [ps.SEX_MALE, ps.SEX_FEMALE, ps.SEX_MALE, ps.SEX_FEMALE]
    assert any("plink" in r.message for r in caplog.records if r.levelno == logging.INFO)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_auto_detect_plink_with_zero_unknown(caplog):
    """PLINK fam with sex={0,1,2} resolves to plink; 0 becomes sentinel."""
    s = pd.Series(["0", "1", "2", "0"])
    with caplog.at_level(logging.INFO, logger="pedigree_summary"):
        out = ps._decode_sex(s, encoding="auto")
    assert out.tolist() == [ps.SEX_UNKNOWN, ps.SEX_MALE, ps.SEX_FEMALE, ps.SEX_UNKNOWN]


def test_auto_detect_zero_as_missing_flips_to_plink():
    """0 tokens + zero_as_missing flips encoding to plink."""
    s = pd.Series(["0", "1", "0", "1"])
    out = ps._decode_sex(s, encoding="auto", zero_as_missing=True)
    # under plink, 0 is missing, 1 is male
    assert out.tolist() == [ps.SEX_UNKNOWN, ps.SEX_MALE, ps.SEX_UNKNOWN, ps.SEX_MALE]


def test_auto_detect_ambiguous_only_ones_warns(caplog):
    """Sex column with only '1' tokens triggers a WARNING."""
    s = pd.Series(["1", "1", "1", "1"])
    with caplog.at_level(logging.WARNING, logger="pedigree_summary"):
        ps._decode_sex(s, encoding="auto")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING for ambiguous only-1 column"
    assert "1" in warnings[0].message


def test_auto_detect_words_only_no_warning(caplog):
    """Word-only sex columns produce NO WARNING (encoding is moot)."""
    s = pd.Series(["M", "F", "Male", "Female"])
    with caplog.at_level(logging.WARNING, logger="pedigree_summary"):
        ps._decode_sex(s, encoding="auto")
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_plink_sex_flag_alias_via_cli(tmp_path):
    """--plink-sex sets sex_encoding=plink without explicit --sex-encoding."""
    ped = _write_ped(tmp_path / "p.tsv", [
        {"id": 1, "sex": "1", "mother": -1, "father": -1},
        {"id": 2, "sex": "2", "mother": -1, "father": -1},
        {"id": 3, "sex": "1", "mother": 2, "father": 1},
    ])
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "out"), "--plink-sex"])
    assert r.returncode == 0, r.stderr


def test_sex_missing_tokens_recognized():
    """Empty / NA / -1 / U / Unknown all decode to SEX_UNKNOWN."""
    s = pd.Series(["M", "F", "", "NA", "-1", "U", "Unknown"])
    out = ps._decode_sex(s, encoding="auto")
    assert out.tolist() == [
        ps.SEX_MALE, ps.SEX_FEMALE,
        ps.SEX_UNKNOWN, ps.SEX_UNKNOWN, ps.SEX_UNKNOWN,
        ps.SEX_UNKNOWN, ps.SEX_UNKNOWN,
    ]


def test_unknown_encoding_raises():
    """Passing an unknown encoding string is a clean error."""
    s = pd.Series(["M", "F"])
    with pytest.raises(ps.PedigreeError, match="unknown sex encoding"):
        ps._decode_sex(s, encoding="bogus")


# ---------------------------------------------------------------------------
# Missing-sex imputation
# ---------------------------------------------------------------------------


def test_missing_sex_imputed_from_mother_role(tmp_path):
    """Row sex='' that is referenced as a mother gets imputed to F."""
    ped = _write_ped(tmp_path / "p.tsv", [
        {"id": 1, "sex": "M", "mother": -1, "father": -1},
        {"id": 2, "sex": "",  "mother": -1, "father": -1},  # missing sex
        {"id": 3, "sex": "F", "mother": 2,  "father": 1},   # uses 2 as mother
    ])
    df, _ = ps.load_and_validate(ped)
    row = df.loc[df["id"] == 2].iloc[0]
    assert int(row["sex"]) == ps.SEX_FEMALE


def test_missing_sex_imputed_from_father_role(tmp_path):
    """Row sex='' that is referenced as a father gets imputed to M."""
    ped = _write_ped(tmp_path / "p.tsv", [
        {"id": 1, "sex": "F", "mother": -1, "father": -1},
        {"id": 2, "sex": "",  "mother": -1, "father": -1},  # missing sex
        {"id": 3, "sex": "M", "mother": 1,  "father": 2},   # uses 2 as father
    ])
    df, _ = ps.load_and_validate(ped)
    row = df.loc[df["id"] == 2].iloc[0]
    assert int(row["sex"]) == ps.SEX_MALE


def test_missing_sex_unresolvable_without_flag_raises(tmp_path):
    """Orphan unsexed row without --allow-missing-sex is an error."""
    ped = _write_ped(tmp_path / "p.tsv", [
        {"id": 1, "sex": "M", "mother": -1, "father": -1},
        {"id": 2, "sex": "F", "mother": -1, "father": -1},
        {"id": 3, "sex": "",  "mother": 2,  "father": 1},   # orphan, no role
    ])
    with pytest.raises(ps.PedigreeError, match="unknown_sex"):
        ps.load_and_validate(ped)


def test_missing_sex_unresolvable_with_flag_keeps_sentinel(tmp_path):
    """allow_missing_sex=True keeps SEX_UNKNOWN sentinel through to df."""
    ped = _write_ped(tmp_path / "p.tsv", [
        {"id": 1, "sex": "M", "mother": -1, "father": -1},
        {"id": 2, "sex": "F", "mother": -1, "father": -1},
        {"id": 3, "sex": "",  "mother": 2,  "father": 1},
    ])
    df, _ = ps.load_and_validate(ped, allow_missing_sex=True)
    assert (df["sex"] == ps.SEX_UNKNOWN).any()


def test_n_unknown_sex_in_size_structure(tmp_path):
    """compute_size_structure exposes n_unknown_sex; n_male + n_female + n_unknown == n."""
    ped = _write_ped(tmp_path / "p.tsv", [
        {"id": 1, "sex": "M", "mother": -1, "father": -1},
        {"id": 2, "sex": "F", "mother": -1, "father": -1},
        {"id": 3, "sex": "",  "mother": 2,  "father": 1},
    ])
    df, csr = ps.load_and_validate(ped, allow_missing_sex=True)
    # ped_depth is populated by _run_summarize from PedigreeGraph; for this
    # unit test fake it from generation order (founders depth 0, kid depth 1).
    df = df.copy()
    df["ped_depth"] = np.where(
        (df["mother"] == -1) & (df["father"] == -1), 0, 1,
    ).astype(np.int32)
    summary, _ = ps.compute_size_structure(df, csr)
    assert summary["n_male"] + summary["n_female"] + summary["n_unknown_sex"] == summary["n_total"]
    assert summary["n_unknown_sex"] == 1


def _ambig_pedigree(path):
    """Pedigree where id=7 is unsexed and used as both mother and father."""
    return _write_ped(path, [
        {"id": 7, "sex": "",  "mother": -1, "father": -1},
        {"id": 8, "sex": "M", "mother": -1, "father": -1},
        {"id": 9, "sex": "F", "mother": -1, "father": -1},
        {"id": 10, "sex": "F", "mother": 7, "father": 8},  # uses 7 as mother
        {"id": 11, "sex": "M", "mother": 9, "father": 7},  # uses 7 as father
    ])


def test_sex_role_ambiguity_raises_in_load_without_flag(tmp_path):
    """Without --allow-missing-sex, ambiguity blocks load_and_validate."""
    ped = _ambig_pedigree(tmp_path / "p.tsv")
    with pytest.raises(ps.PedigreeError, match="sex_role_ambiguity"):
        ps.load_and_validate(ped)


def test_load_and_validate_allows_sex_ambiguity_with_flag(tmp_path):
    """With allow_missing_sex=True, ambiguous row passes with sex==SEX_UNKNOWN."""
    ped = _ambig_pedigree(tmp_path / "p.tsv")
    df, _ = ps.load_and_validate(ped, allow_missing_sex=True)
    row = df.loc[df["id"] == 7].iloc[0]
    assert int(row["sex"]) == ps.SEX_UNKNOWN


# ---------------------------------------------------------------------------
# Hard-refuse: --allow-missing-sex + --effective-size / --inbreeding
# ---------------------------------------------------------------------------


def test_unknown_sex_blocks_effective_size_in_summarize(tmp_path):
    """Summarize --allow-missing-sex --effective-size exits 1 with clear message."""
    ped = _write_ped(tmp_path / "p.tsv", [
        {"id": 1, "sex": "M", "mother": -1, "father": -1},
        {"id": 2, "sex": "F", "mother": -1, "father": -1},
        {"id": 3, "sex": "",  "mother": 2,  "father": 1},
    ])
    r = run_pedsum([
        "summarize", "--in", str(ped), "--out", str(tmp_path / "s"),
        "--allow-missing-sex", "--effective-size",
    ])
    assert r.returncode == 1, f"expected exit 1, got {r.returncode}\nstderr:\n{r.stderr}"
    assert "sex-stratified" in r.stderr.lower() or "resolved sex" in r.stderr.lower()


def test_unknown_sex_blocks_inbreeding_in_summarize(tmp_path):
    """Summarize --allow-missing-sex --inbreeding exits 1 with clear message."""
    ped = _write_ped(tmp_path / "p.tsv", [
        {"id": 1, "sex": "M", "mother": -1, "father": -1},
        {"id": 2, "sex": "F", "mother": -1, "father": -1},
        {"id": 3, "sex": "",  "mother": 2,  "father": 1},
    ])
    r = run_pedsum([
        "summarize", "--in", str(ped), "--out", str(tmp_path / "s"),
        "--allow-missing-sex", "--inbreeding",
    ])
    assert r.returncode == 1, f"expected exit 1, got {r.returncode}\nstderr:\n{r.stderr}"
    assert "sex-stratified" in r.stderr.lower() or "resolved sex" in r.stderr.lower()
