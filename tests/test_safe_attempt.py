"""Coverage for the ``--safe-attempt`` small-cell redaction path.

The subprocess test pins the CLI contract (no annotated.tsv.gz written,
INFO lines on stderr). The direct unit test pins the headline branches
of ``_apply_safe_attempt`` — size_structure / pairs / sex_summary /
inbreeding / ind_data.distributions. Sections deliberately not covered:
family_size, mating_pairs, lineage, founder_contribution,
founder_generation, components, generation_summary.
"""

from __future__ import annotations

import pedigree_summary as ps
from conftest import run_pedsum
from conftest import write_ped as _write_ped


def test_safe_attempt_summarize_outputs(tmp_path):
    """``summarize --safe-attempt`` skips annotated TSV and logs the redaction."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "M", "mother": 2, "father": 1},
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
            "--safe-attempt",
            "--no-inbreeding",
            "--no-effective-size",
        ]
    )
    assert r.returncode == 0, r.stderr
    assert (out_dir / "summary.yaml").exists()
    assert (out_dir / "summary.extra.yaml").exists()
    assert not (out_dir / "annotated.tsv.gz").exists()
    # --tsv was not requested, so the long-form TSVs stay absent regardless.
    assert not (out_dir / "summary.pedigree.tsv").exists()
    assert not (out_dir / "summary.individual.tsv").exists()
    assert "safe-attempt redaction applied" in r.stderr
    assert "safe-attempt: skipped" in r.stderr


def test_apply_safe_attempt_headline_branches():
    """Direct in-process call covers the four headline redaction branches."""
    ped_data = {
        "n_total": 100,
        "size_structure": {
            "next_components": [3, 7],
            "gen_counts": [2, 6],
            "largest_component": 4,
        },
        "pairs": {"by_degree": {"1": 3, "2": 12}, "FS": 2},
        "sex_summary": {
            "M": {"n": 3, "n_founders": 3},
            "F": {"n": 10, "n_founders": 2},
        },
        "inbreeding": {
            "n_inbred": 2,
            "frac_inbred": 0.02,
            "mean_F": 0.01,
            "max_F": 0.05,
            "hist": {"(0,0.05]": 0.03, "(0.05,0.1]": 0.5},
        },
    }
    ind_data = {
        "distributions": {
            "n_children": {"min": 0, "max": 5, "nz": 2},
            "kinship": {"min": 0.0, "max": 0.5, "nz": 7},
        },
    }

    ps._apply_safe_attempt(ped_data, ind_data, min_cell=5)

    # size_structure: small components dropped; small gen counts nulled;
    # small largest_component nulled.
    assert ped_data["size_structure"]["next_components"] == [7]
    assert ped_data["size_structure"]["gen_counts"] == [None, 6]
    assert ped_data["size_structure"]["largest_component"] is None

    # pairs: small by_degree buckets and small bare counts nulled.
    assert ped_data["pairs"]["by_degree"]["1"] is None
    assert ped_data["pairs"]["by_degree"]["2"] == 12
    assert ped_data["pairs"]["FS"] is None

    # sex_summary: M (n=3 < min_cell) nulls everything except n;
    # F (n=10) enters the else branch and nulls n_founders=2 (< min_cell).
    assert ped_data["sex_summary"]["M"]["n"] == 3
    assert ped_data["sex_summary"]["M"]["n_founders"] is None
    assert ped_data["sex_summary"]["F"]["n"] == 10
    assert ped_data["sex_summary"]["F"]["n_founders"] is None

    # inbreeding: scalar quartet nulled because n_inbred < min_cell;
    # hist buckets suppressed only when frac * n_total < min_cell.
    inb = ped_data["inbreeding"]
    assert inb["n_inbred"] is None
    assert inb["frac_inbred"] is None
    assert inb["mean_F"] is None
    assert inb["max_F"] is None
    assert inb["hist"]["(0,0.05]"] is None  # 0.03 * 100 = 3 < 5
    assert inb["hist"]["(0.05,0.1]"] == 0.5  # 0.50 * 100 = 50 ≥ 5

    # ind_data.distributions: min/max dropped unconditionally; nz nulled
    # only when below min_cell.
    n_children = ind_data["distributions"]["n_children"]
    kinship = ind_data["distributions"]["kinship"]
    assert "min" not in n_children
    assert "max" not in n_children
    assert "min" not in kinship
    assert "max" not in kinship
    assert n_children["nz"] is None  # 2 < 5
    assert kinship["nz"] == 7  # 7 ≥ 5
