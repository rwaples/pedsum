"""Behavior tests for the validate subcommand: hard-blocks, summary layout."""

from __future__ import annotations

import gzip

import pandas as pd
from conftest import run_pedsum
from conftest import write_ped as _write_ped


def test_validate_unknown_sex_is_hard_block(tmp_path):
    """Orphan unsexed row without --allow-missing-sex hard-blocks validate."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "", "mother": 2, "father": 1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 2, r.stderr
    assert "BLOCKED" in r.stderr
    assert "unresolved sex" in r.stderr.lower() or "unknown" in r.stderr.lower()
    # .validate.tsv.gz must NOT be written when blocked
    assert not (tmp_path / "out" / "validate.tsv.gz").exists()


def test_validate_unknown_sex_skipped_with_flag(tmp_path):
    """--allow-missing-sex turns the unknown_sex check into a SKIP."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "", "mother": 2, "father": 1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(base),
            "--allow-missing-sex",
        ]
    )
    assert r.returncode == 0, r.stderr
    # Summary shows SKIP for unknown_sex with the tolerated count.
    assert "SKIP" in r.stderr
    assert "tolerated" in r.stderr
    # Output file IS written.
    assert (tmp_path / "out" / "validate.tsv.gz").exists()
    # validate.log does NOT contain rows for tolerated unknown_sex findings
    # (matches existing precedent: _check_* is skipped entirely when flagged).
    log = (tmp_path / "out" / "validate.log").read_text()
    assert "unknown_sex" not in log


def _ambig_pedigree(path):
    """Pedigree where id=7 is unsexed and used as both mother and father."""
    return _write_ped(
        path,
        [
            {"id": 7, "sex": "", "mother": -1, "father": -1},
            {"id": 8, "sex": "M", "mother": -1, "father": -1},
            {"id": 9, "sex": "F", "mother": -1, "father": -1},
            {"id": 10, "sex": "F", "mother": 7, "father": 8},
            {"id": 11, "sex": "M", "mother": 9, "father": 7},
        ],
    )


def test_validate_sex_role_ambiguity_is_hard_block_without_flag(tmp_path):
    """sex_role_ambiguity hard-blocks validate when --allow-missing-sex is absent."""
    ped = _ambig_pedigree(tmp_path / "p.tsv")
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 2, r.stderr
    assert "BLOCKED" in r.stderr
    assert not (tmp_path / "out" / "validate.tsv.gz").exists()


def test_validate_sex_role_ambiguity_passes_with_flag(tmp_path):
    """--allow-missing-sex lets ambiguity through; fixed TSV writes sex=-1 for the row."""
    import gzip

    import pandas as pd

    ped = _ambig_pedigree(tmp_path / "p.tsv")
    base = tmp_path / "out"
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(base),
            "--allow-missing-sex",
        ]
    )
    assert r.returncode == 0, r.stderr
    # SKIP message in the grouped summary; no BLOCKED.
    assert "BLOCKED" not in r.stderr
    assert "tolerated" in r.stderr
    fixed = tmp_path / "out" / "validate.tsv.gz"
    assert fixed.exists()
    with gzip.open(fixed, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str)
    row = df.loc[df["id"].astype(int) == 7].iloc[0]
    assert row["sex"] == "-1"
    # validate.log should NOT contain a row for the tolerated ambiguous id.
    log = (tmp_path / "out" / "validate.log").read_text()
    assert "sex_role_ambiguity" not in log


def test_orphan_unsexed_writes_minus_one_in_fixed_tsv(tmp_path):
    """Orphan-only pedigree (n_imputed==0) + --allow-missing-sex: orphan sex normalised to -1."""
    import gzip

    import pandas as pd

    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "", "mother": 2, "father": 1},  # orphan, no role
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(base),
            "--allow-missing-sex",
        ]
    )
    assert r.returncode == 0, r.stderr
    fixed = tmp_path / "out" / "validate.tsv.gz"
    assert fixed.exists()
    with gzip.open(fixed, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str)
    row = df.loc[df["id"].astype(int) == 3].iloc[0]
    assert row["sex"] == "-1"


def test_validate_summary_uses_grouped_layout(tmp_path):
    """The validate stderr summary prints the four section headers in order."""
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
    expected = [
        "Columns & parsing",
        "IDs",
        "Parent references",
        "Graph structure",
    ]
    last = -1
    for hdr in expected:
        idx = r.stderr.find(hdr)
        assert idx > last, f"section header {hdr!r} missing or out of order"
        last = idx
    # Dot-padded labels: lines like "required columns present .... PASS"
    assert ".. PASS" in r.stderr


def test_validate_writes_imputed_sex_in_fixed_output(tmp_path):
    """The .validate.tsv.gz contains the imputed sex value (F/M) not the blank."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "", "mother": -1, "father": -1},  # imputed F (used as mother)
            {"id": 3, "sex": "F", "mother": 2, "father": 1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 0, r.stderr
    with gzip.open(tmp_path / "out" / "validate.tsv.gz", "rt") as fh:
        fixed = pd.read_csv(fh, sep="\t", dtype=str)
    row2 = fixed.loc[fixed["id"].astype(int) == 2].iloc[0]
    assert row2["sex"] == "F"
