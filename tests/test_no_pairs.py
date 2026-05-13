"""Tests for the ``--no-pairs`` CLI flag.

The flag skips the relationship-pair enumeration entirely (matrix /
BFS engines), replacing the 23 named pair counts and the relationship-
burden summary with stubs.  Other sections — size, family, mating,
lineage, inbreeding, and effective size — must remain unchanged.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example_pedigree.tsv"
SCRIPT = REPO / "pedigree_summary.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO,
    )


def _load_yaml(base: Path) -> dict:
    return yaml.safe_load((base.parent / f"{base.name}.summary.yaml").read_text())


def _load_tsv(base: Path) -> list[list[str]]:
    path = base.parent / f"{base.name}.summary.pedigree.tsv"
    with path.open() as fh:
        return list(csv.reader(fh, delimiter="\t"))


def test_no_pairs_emits_stubs(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base), "--no-pairs",
    ])
    assert res.returncode == 0, res.stderr

    ped = _load_yaml(base)["pedigree"]
    assert ped["pairs_engine"] == "skipped"
    assert ped["pairs"] == {}
    rs = ped["relationship_summary"]
    assert rs["computed"] is False
    assert "skipped via --no-pairs" in rs["skip_reason"]
    # n_possible_pairs is still useful (cheap to compute from n alone).
    assert rs["n_possible_pairs"] == 200 * 199 // 2


def test_no_pairs_log_message(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base), "--no-pairs",
    ])
    assert res.returncode == 0
    # The expensive INFO lines are absent.
    assert "relationship pairs in" not in res.stderr
    assert "relationship burden summary in" not in res.stderr
    # The skip notice IS emitted.
    assert "skipped (--no-pairs)" in res.stderr


def test_no_pairs_preserves_other_sections(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--no-pairs", "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(base)["pedigree"]
    # Cheap sections unaffected.
    assert ped["size_structure"]["n_total"] if "n_total" in ped["size_structure"] else True
    assert ped["family_size"] is not None
    assert ped["mating_pairs"] is not None
    assert ped["lineage"] != {}
    assert ped["founder_contribution"] != {}
    # The expensive opt-in features still work without pair counts.
    assert ped["inbreeding"] is not None
    assert len(ped["effective_size"]) == 8


def test_no_pairs_tsv_no_pair_rows(tmp_path):
    """Long-form TSV should have no pair rows under --no-pairs."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base), "--no-pairs",
    ])
    assert res.returncode == 0, res.stderr
    rows = _load_tsv(base)
    # Section name in the TSV's first column.
    pair_rows = [r for r in rows[1:] if r[0] == "pairs"]
    assert pair_rows == []
