"""Tests for the two pair-counting modes in ``summarize``.

The default mode uses ``pg.count_pairs_streaming`` (scalar, O(N)
memory; bit-identical to the matrix engine for the 10 simple codes,
~1% approximate for the 13 cousin / collateral codes on inbred input).
``--per-individual-pairs`` opts into the matrix engine to populate the
per-individual relationship-burden summary, at the cost of OOM risk
on pair-dense pedigrees.
"""

from __future__ import annotations

from conftest import EXAMPLE
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run

# ----- default (streaming scalar) mode --------------------------------


def test_default_uses_streaming_engine(tmp_path):
    """Default summarize routes through the streaming scalar pair-count engine."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(out_dir)["pedigree"]
    assert ped["relatedness"]["pairs"]["engine"] == "streaming_scalar"


def test_default_populates_pair_counts(tmp_path):
    """Streaming engine emits all 23 named codes plus PO synthesis; by_degree dropped from YAML."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    pairs = _load_yaml(out_dir)["pedigree"]["relatedness"]["pairs"]
    # 23 named codes + PO synthesis.
    for code in ("MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "GGP", "GGGP", "G3GP", "Av", "HAv", "1C", "H1C", "2C"):
        assert code in pairs, f"{code} missing from default pairs dict"
    assert "PO" in pairs
    # by_degree is a YAML-only drop (derivable from the 23 codes; still in TSV).
    assert "by_degree" not in pairs
    assert pairs["MO"] > 0
    assert pairs["FS"] > 0


def test_default_relationship_summary_is_stub(tmp_path):
    """In streaming mode ``relationship_summary`` is a stub with skip_reason set."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    rs = _load_yaml(out_dir)["pedigree"]["relatedness"]["relationship_summary"]
    assert rs["computed"] is False
    assert "pass --per-individual-pairs" in rs["skip_reason"]
    assert rs["n_possible_pairs"] == 200 * 199 // 2


def test_default_works_with_inbreeding_and_effective_size(tmp_path):
    """Streaming engine composes with ``--inbreeding`` and ``--effective-size`` (now defaults)."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(out_dir)["pedigree"]
    assert ped["relatedness"]["pairs"]["engine"] == "streaming_scalar"
    assert ped["relatedness"]["inbreeding"] is not None
    assert len(ped["popgen"]["effective_size"]) == 8


# ----- --per-individual-pairs (matrix engine) mode --------------------


def test_per_individual_pairs_uses_matrix_engine(tmp_path):
    """``--per-individual-pairs`` routes through the matrix pair-count engine."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--per-individual-pairs",
        ]
    )
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(out_dir)["pedigree"]
    # engine field comes from count_relationship_pairs.
    assert ped["relatedness"]["pairs"]["engine"] in ("matrix", "bfs")


def test_per_individual_pairs_populates_relationship_summary(tmp_path):
    """``--per-individual-pairs`` populates the per-individual relationship-burden summary."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--per-individual-pairs",
        ]
    )
    assert res.returncode == 0, res.stderr
    rs = _load_yaml(out_dir)["pedigree"]["relatedness"]["relationship_summary"]
    assert rs["computed"] is True
    # Per-individual burden fields appear when matrix engine ran.
    assert "relatives_by_degree" in rs
    assert "n_related_pairs" in rs


def test_per_individual_pairs_counts_close_to_streaming(tmp_path):
    """On the example pedigree, the two modes give identical counts for the 10 exact codes."""
    out_s = tmp_path / "stream"
    out_b = tmp_path / "burden"
    assert _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_s)]).returncode == 0
    assert (
        _run(
            [
                "summarize",
                "--in",
                str(EXAMPLE),
                "--out",
                str(out_b),
                "--per-individual-pairs",
            ]
        ).returncode
        == 0
    )
    s = _load_yaml(out_s)["pedigree"]["relatedness"]["pairs"]
    b = _load_yaml(out_b)["pedigree"]["relatedness"]["pairs"]
    # The 10 simple codes match bit-identically.
    for code in ("MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "GGP", "GGGP", "G3GP"):
        assert s[code] == b[code], f"{code}: streaming={s[code]} burden={b[code]}"
