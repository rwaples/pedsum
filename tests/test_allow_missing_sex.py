"""``--allow-missing-sex`` on ``summarize``: unsexed rows are kept, not refused.

A role-ambiguous row and an orphan row both finish with PLINK sex 0, and
inbreeding on a pedigree with unsexed rows matches the fully sexed run.
"""

from __future__ import annotations

import csv

import polars as pl
from conftest import load_annotated_tsv_gz
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run


def _ambiguous_and_orphan_pedigree(path):
    """A pedigree with both a role-ambiguous row (id=7) AND an orphan (id=8)."""
    rows = [
        {"id": 1, "sex": "M", "mother": -1, "father": -1},
        {"id": 2, "sex": "F", "mother": -1, "father": -1},
        {"id": 7, "sex": "", "mother": -1, "father": -1},  # ambiguous below
        {"id": 8, "sex": "", "mother": -1, "father": -1},  # orphan
        {"id": 3, "sex": "M", "mother": 7, "father": 1},
        {"id": 4, "sex": "F", "mother": 2, "father": 7},
    ]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return path


def test_summarize_missing_sex_keeps_unsexed_rows(tmp_path):
    """``--allow-missing-sex`` with inbreeding on finishes on a role-ambiguous and an orphan row."""
    ped = _ambiguous_and_orphan_pedigree(tmp_path / "ped.tsv")
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(ped), "--out", str(out_dir), "--allow-missing-sex"])
    assert res.returncode == 0, res.stderr
    ann = load_annotated_tsv_gz(out_dir)
    for orig_id in (7, 8):
        row = ann.filter(pl.col("id") == orig_id).row(0, named=True)
        assert int(row["sex"]) == 0, f"id={orig_id} sex={row['sex']}"  # PLINK unknown


def _inbred_pedigree(path, unknown_sex: str):
    """Full sibs 3 x 4 have child 5 (F = 0.25); 5 and orphan 8 carry ``unknown_sex``."""
    rows = [
        {"id": 1, "sex": "M", "mother": -1, "father": -1},
        {"id": 2, "sex": "F", "mother": -1, "father": -1},
        {"id": 3, "sex": "M", "mother": 2, "father": 1},
        {"id": 4, "sex": "F", "mother": 2, "father": 1},
        {"id": 5, "sex": unknown_sex, "mother": 4, "father": 3},
        {"id": 8, "sex": unknown_sex, "mother": -1, "father": -1},
    ]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return path


def test_summarize_inbreeding_does_not_need_resolved_sex(tmp_path):
    """F on a pedigree with unsexed rows equals F with their sex filled in (ADR 0004).

    0.14 refused ``--allow-missing-sex`` with inbreeding on; F never read sex.
    """
    unsexed = tmp_path / "unsexed"
    res = _run(
        [
            "summarize",
            "--in",
            str(_inbred_pedigree(tmp_path / "u.tsv", "")),
            "--out",
            str(unsexed),
            "--allow-missing-sex",
        ]
    )
    assert res.returncode == 0, res.stderr
    sexed = tmp_path / "sexed"
    res = _run(["summarize", "--in", str(_inbred_pedigree(tmp_path / "s.tsv", "F")), "--out", str(sexed)])
    assert res.returncode == 0, res.stderr

    inbreeding = _load_yaml(unsexed)["pedigree"]["relatedness"]["inbreeding"]
    assert inbreeding == _load_yaml(sexed)["pedigree"]["relatedness"]["inbreeding"]
    assert inbreeding["n_inbred"] == 1
    assert inbreeding["max_F"] == 0.25
    assert "sex_summary" in _load_yaml(unsexed)["pedigree"]["strata"]
    ann = load_annotated_tsv_gz(unsexed)
    assert ann.filter(pl.col("id").is_in([5, 8]))["sex_source"].to_list() == ["unresolved", "unresolved"]
    assert "--allow-unknown-sex" not in res.stderr
