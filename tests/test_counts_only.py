"""Tests for the ``--counts-only`` CLI flag.

The flag uses ``pedigree_graph.PedigreeGraph.count_pairs_streaming``
(scalar / memory-bounded) to populate the 23 pair counts.  The
relationship-burden summary stays a stub (per-individual cousin /
aunt-uncle counts need full pair-list enumeration).  Other sections —
size, family, mating, lineage, inbreeding, effective size — work as
usual.
"""
from __future__ import annotations

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


def test_counts_only_populates_pairs(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base), "--counts-only",
    ])
    assert res.returncode == 0, res.stderr

    ped = _load_yaml(base)["pedigree"]
    assert ped["pairs_engine"] == "streaming_scalar"
    pairs = ped["pairs"]
    # 23 named codes + PO synthesis + by_degree rollup.
    assert "MZ" in pairs and "FS" in pairs and "2C" in pairs
    assert "PO" in pairs   # synthesized via _augment_pair_counts
    assert "by_degree" in pairs
    # Counts are integers and at least the lineal/sibling ones are positive.
    assert pairs["MO"] > 0
    assert pairs["FS"] > 0


def test_counts_only_relationship_summary_is_stub(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base), "--counts-only",
    ])
    assert res.returncode == 0, res.stderr
    rs = _load_yaml(base)["pedigree"]["relationship_summary"]
    assert rs["computed"] is False
    assert "per-individual relationship burden" in rs["skip_reason"]
    assert rs["n_possible_pairs"] == 200 * 199 // 2


def test_counts_only_with_inbreeding_and_effective_size(tmp_path):
    """Other opt-in sections still work alongside --counts-only."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--counts-only", "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(base)["pedigree"]
    assert ped["pairs_engine"] == "streaming_scalar"
    assert ped["inbreeding"] is not None
    assert len(ped["effective_size"]) == 8


def test_counts_only_no_pairs_mutually_exclusive(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--counts-only", "--no-pairs",
    ])
    assert res.returncode == 1
    assert "mutually exclusive" in res.stderr
