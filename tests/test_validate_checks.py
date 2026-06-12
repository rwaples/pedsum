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


def test_sex_role_consistency_log_pins_offending_rows(tmp_path):
    """sex_role_consistency names the lines, not just the id.

    id=3 (row 2, female) is used as a father in rows 4 and 5. The finding now
    carries the id's own row in the ``row`` column and the referencing rows in
    the detail, so the log is actionable without grepping the input.
    """
    import pandas as pd

    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},
            {"id": 3, "sex": "F", "mother": -1, "father": -1},  # row 2
            {"id": 4, "sex": "F", "mother": 3, "father": 2},  # 3 as mother (consistent)
            {"id": 5, "sex": "M", "mother": 1, "father": 3},  # row 4: 3 as father
            {"id": 6, "sex": "M", "mother": 1, "father": 3},  # row 5: 3 as father
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 1, r.stderr  # FAIL but not a hard block
    log = pd.read_csv(base / "validate.log", sep="\t")
    finding = log[log["check"] == "sex_role_consistency"]
    assert len(finding) == 1
    assert int(finding.iloc[0]["id"]) == 3
    assert int(finding.iloc[0]["row"]) == 2  # id=3's own row (where its sex lives)
    assert "row(s) [4, 5]" in finding.iloc[0]["detail"]


def test_parent_refs_sex_conflict_log_pins_rows(tmp_path):
    """parent_refs_sex_conflict names the mother and father referencing rows.

    Missing parent id=99 is referenced as mother in rows 2 and 4 and as father
    in row 3; the detail now names both sets of lines.
    """
    import pandas as pd

    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},
            {"id": 4, "sex": "F", "mother": 99, "father": 2},  # row 2: 99 as mother
            {"id": 5, "sex": "M", "mother": 1, "father": 99},  # row 3: 99 as father
            {"id": 7, "sex": "F", "mother": 99, "father": 2},  # row 4: 99 as mother
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 2, r.stderr  # hard block
    log = pd.read_csv(base / "validate.log", sep="\t")
    finding = log[log["check"] == "parent_refs_sex_conflict"]
    assert len(finding) == 1
    detail = finding.iloc[0]["detail"]
    assert "mother (row(s) [2, 4])" in detail
    assert "father (row(s) [3])" in detail


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
