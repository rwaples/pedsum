"""``--allow-missing-sex`` on ``summarize``: unsexed rows are kept, not refused.

An orphan row finishes with PLINK sex 0, and inbreeding on a pedigree with
unsexed rows matches the fully sexed run. A role-ambiguous row is kept by
validate but refused by the commands that build a pedigree graph.
"""

from __future__ import annotations

import csv

import polars as pl
import pytest
from conftest import load_annotated_tsv_gz, load_validate_tsv_gz
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run


def _ambiguous_pedigree(path):
    """Unsexed ``u7`` is the mother of ``c3`` and the father of ``c4``; ``u8`` is an unsexed orphan."""
    rows = [
        {"id": "m1", "sex": "M", "mother": -1, "father": -1},
        {"id": "f2", "sex": "F", "mother": -1, "father": -1},
        {"id": "u7", "sex": "", "mother": -1, "father": -1},
        {"id": "u8", "sex": "", "mother": -1, "father": -1},
        {"id": "c3", "sex": "M", "mother": "u7", "father": "m1"},
        {"id": "c4", "sex": "F", "mother": "f2", "father": "u7"},
    ]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return path


def test_validate_keeps_an_unsexed_parent_in_both_roles(tmp_path):
    """Validate --allow-missing-sex still keeps the role-ambiguous row, written with sex 0."""
    out_dir = tmp_path / "out"
    res = _run(
        ["validate", "--in", str(_ambiguous_pedigree(tmp_path / "p.tsv")), "--out", str(out_dir), "--allow-missing-sex"]
    )
    assert res.returncode == 0, res.stderr
    written = load_validate_tsv_gz(out_dir)
    assert written.filter(pl.col("id") == "u7")["sex"].item() == "0"


@pytest.mark.parametrize("command", ["summarize", "effective-size", "epimight-input"])
def test_graph_commands_refuse_an_unsexed_parent_in_both_roles(tmp_path, command):
    """pedigree-graph 0.15 refuses an id in both parent roles; the commands say which and how to repair it."""
    ped = _ambiguous_pedigree(tmp_path / "p.tsv")
    extra = ["--estimators", "ne_inbreeding"] if command == "effective-size" else []
    res = _run([command, "--in", str(ped), "--out", str(tmp_path / "out"), "--allow-missing-sex", *extra])
    assert res.returncode == 1, res.stderr
    assert "id u7 is the mother of id c3 and the father of id c4" in res.stderr
    assert "Rerun validate without them" in res.stderr
    assert "Traceback" not in res.stderr


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
