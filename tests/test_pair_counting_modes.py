"""Tests for the two pair-counting modes in ``summarize``.

The default mode uses ``pg.count_pairs_streaming`` (scalar, O(N)
memory; bit-identical to the matrix engine for the 10 simple codes,
~1% approximate for the 13 cousin / collateral codes on inbred input).
``--burden`` opts into the matrix / BFS engine to populate the
per-individual relationship-burden summary, at the cost of OOM risk
on pair-dense pedigrees.
"""
from __future__ import annotations

from pathlib import Path

from conftest import EXAMPLE, load_summary_yaml as _load_yaml, run_pedsum as _run


# ----- default (streaming scalar) mode --------------------------------


def test_default_uses_streaming_engine(tmp_path):
    base = tmp_path / "p"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(base)])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(base)["pedigree"]
    assert ped["pairs_engine"] == "streaming_scalar"


def test_default_populates_pair_counts(tmp_path):
    base = tmp_path / "p"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(base)])
    assert res.returncode == 0, res.stderr
    pairs = _load_yaml(base)["pedigree"]["pairs"]
    # 23 named codes + PO synthesis + by_degree rollup.
    for code in ("MZ", "MO", "FO", "FS", "MHS", "PHS",
                 "GP", "GGP", "GGGP", "G3GP", "Av", "HAv",
                 "1C", "H1C", "2C"):
        assert code in pairs, f"{code} missing from default pairs dict"
    assert "PO" in pairs
    assert "by_degree" in pairs
    assert pairs["MO"] > 0
    assert pairs["FS"] > 0


def test_default_relationship_summary_is_stub(tmp_path):
    base = tmp_path / "p"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(base)])
    assert res.returncode == 0, res.stderr
    rs = _load_yaml(base)["pedigree"]["relationship_summary"]
    assert rs["computed"] is False
    assert "pass --burden" in rs["skip_reason"]
    assert rs["n_possible_pairs"] == 200 * 199 // 2


def test_default_works_with_inbreeding_and_effective_size(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(base)["pedigree"]
    assert ped["pairs_engine"] == "streaming_scalar"
    assert ped["inbreeding"] is not None
    assert len(ped["effective_size"]) == 8


# ----- --burden (matrix engine) mode ----------------------------------


def test_burden_uses_matrix_engine(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base), "--burden",
    ])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(base)["pedigree"]
    # _engine field comes from count_relationship_pairs, set to "matrix" or "bfs".
    assert ped["pairs_engine"] in ("matrix", "bfs")


def test_burden_populates_relationship_summary(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base), "--burden",
    ])
    assert res.returncode == 0, res.stderr
    rs = _load_yaml(base)["pedigree"]["relationship_summary"]
    assert rs["computed"] is True
    # Per-individual burden fields appear when matrix engine ran.
    assert "relatives_by_degree" in rs
    assert "n_related_pairs" in rs


def test_burden_pair_counts_close_to_streaming(tmp_path):
    """On the example pedigree, the two modes should give similar
    counts for the 10 exact codes."""
    base_s = tmp_path / "stream"
    base_b = tmp_path / "burden"
    assert _run(["summarize", "--in", str(EXAMPLE), "--out", str(base_s)]).returncode == 0
    assert _run(["summarize", "--in", str(EXAMPLE), "--out", str(base_b), "--burden"]).returncode == 0
    s = _load_yaml(base_s)["pedigree"]["pairs"]
    b = _load_yaml(base_b)["pedigree"]["pairs"]
    # The 10 simple codes match bit-identically.
    for code in ("MZ", "MO", "FO", "FS", "MHS", "PHS",
                 "GP", "GGP", "GGGP", "G3GP"):
        assert s[code] == b[code], f"{code}: streaming={s[code]} burden={b[code]}"
