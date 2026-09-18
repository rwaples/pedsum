"""Tests for the two pair-counting modes in ``summarize``.

The default mode uses ``pg.relationship_counts`` (the Rust row-streaming
engine: exact for all 23 codes in O(N) memory). ``--per-individual-pairs``
opts into the matrix engine to populate the per-individual
relationship-burden summary, at the cost of OOM risk on pair-dense
pedigrees. Both modes report the same 23 counts.

Both engines assign each pair its single closest relationship category,
so a pedigree with parent-offspring incest reports the parent-offspring
pair only under ``MO`` / ``FO`` and not additionally as a half sib.
"""

from __future__ import annotations

import numpy as np
from conftest import EXAMPLE, write_ped
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run
from pedigree_graph import RELATIONSHIPS, PedigreeGraph

# Parent-offspring incest: id2 (a founder) also fathers a child on his own
# daughter id5, so the (id5, id7) pair is simultaneously mother-offspring and
# paternal half sib. Two further paternal half-sib pairs (id5/id6, id6/id7)
# carry no such overlap, so the folded count is 2 where the unfolded count
# would be 3 — the fold has to bite without emptying the code.
_INCEST_PEDIGREE = [
    {"id": 1, "sex": "F", "mother": -1, "father": -1},
    {"id": 2, "sex": "M", "mother": -1, "father": -1},
    {"id": 3, "sex": "F", "mother": -1, "father": -1},
    {"id": 4, "sex": "M", "mother": -1, "father": -1},
    {"id": 5, "sex": "F", "mother": 1, "father": 2},
    {"id": 6, "sex": "M", "mother": 3, "father": 2},
    {"id": 7, "sex": "M", "mother": 5, "father": 2},
    {"id": 8, "sex": "M", "mother": 5, "father": 4},
]

#: Every registry code, in registry order: the default engine is exact on all
#: of them, so the two modes must agree code for code.
_ALL_CODES = tuple(RELATIONSHIPS)

# ----- default (Rust row-streaming) mode ------------------------------


def test_default_uses_streaming_engine(tmp_path):
    """Default summarize routes through the Rust row-streaming count engine."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(out_dir)["pedigree"]
    assert ped["relatedness"]["relationship_pairs"]["engine"] == "rust_streaming"
    assert "clamped" not in ped["relatedness"]["relationship_pairs"]


def test_default_populates_pair_counts(tmp_path):
    """Streaming engine emits all 23 named codes plus PO synthesis; by_degree dropped from YAML."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    pairs = _load_yaml(out_dir)["pedigree"]["relatedness"]["relationship_pairs"]
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
    assert rs["n_individual_pairs"] == 200 * 199 // 2


def test_default_works_with_inbreeding_and_effective_size(tmp_path):
    """Streaming engine composes with ``--inbreeding`` and ``--effective-size`` (now defaults)."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(out_dir)["pedigree"]
    assert ped["relatedness"]["relationship_pairs"]["engine"] == "rust_streaming"
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
    # --per-individual-pairs uses the matrix pair-list enumerator.
    assert ped["relatedness"]["relationship_pairs"]["engine"] == "matrix"


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


def test_per_individual_pairs_counts_equal_streaming(tmp_path):
    """On the example pedigree, the two modes report identical counts for all 23 codes and PO."""
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
    s = _load_yaml(out_s)["pedigree"]["relatedness"]["relationship_pairs"]
    b = _load_yaml(out_b)["pedigree"]["relatedness"]["relationship_pairs"]
    for code in (*_ALL_CODES, "PO"):
        assert s[code] == b[code], f"{code}: streaming={s[code]} burden={b[code]}"


# ----- the exact-code contract and the closest-category fold ----------


def test_relationship_counts_are_exact_on_every_code():
    """``relationship_counts(max_degree=5)`` reports every registry code as exact."""
    pg = PedigreeGraph.from_arrays(
        ids=np.arange(4),
        mother_ids=np.array([-1, -1, 0, 0]),
        father_ids=np.array([-1, -1, 1, 1]),
    )
    counts = pg.relationship_counts(max_degree=5)
    assert set(counts.exact) == set(_ALL_CODES)
    assert set(counts.requested) == set(_ALL_CODES)


def test_incest_fold_agrees_between_engines(tmp_path):
    """On parent-offspring incest, both engines fold the overlap the same way."""
    ped = write_ped(tmp_path / "incest.tsv", _INCEST_PEDIGREE)
    out_s = tmp_path / "stream"
    out_b = tmp_path / "burden"
    assert _run(["summarize", "--in", str(ped), "--out", str(out_s)]).returncode == 0
    assert _run(["summarize", "--in", str(ped), "--out", str(out_b), "--per-individual-pairs"]).returncode == 0
    s = _load_yaml(out_s)["pedigree"]["relatedness"]["relationship_pairs"]
    b = _load_yaml(out_b)["pedigree"]["relatedness"]["relationship_pairs"]
    for code in _ALL_CODES:
        assert s[code] == b[code], f"{code}: streaming={s[code]} burden={b[code]}"
    # id5 is both the mother and the paternal half sib of id7; the pair is
    # counted once, as parent-offspring. Three unfolded PHS pairs, two folded.
    assert s["PHS"] == 2
    assert s["MO"] == 4


def test_incest_fold_matches_per_individual_pair_lists(tmp_path):
    """The folded counts equal the pair lists the matrix engine materialises."""
    from pedsum.pairs import _build_pedigree_graph, _count_pairs_matrix_with_lists
    from pedsum.validate import load_and_validate

    ped = write_ped(tmp_path / "incest.tsv", _INCEST_PEDIGREE)
    df, _ = load_and_validate(ped)
    pg = _build_pedigree_graph(df)
    counts = _count_pairs_matrix_with_lists(df, pg=pg)
    streamed = pg.relationship_counts(max_degree=5)
    for code in _ALL_CODES:
        n_pairs = len(counts["_pair_lists"][code])
        assert counts[code] == n_pairs
        assert streamed[code] == n_pairs, f"{code}: streamed={streamed[code]} pairs={n_pairs}"

    # The overlapping pair appears under MO, and under no half-sib code.
    ids = df["id"].to_numpy()
    mo_block = counts["_pair_lists"]["MO"]
    mo_pairs = set(zip(ids[mo_block.first_rows].tolist(), ids[mo_block.second_rows].tolist(), strict=True))
    assert (7, 5) in mo_pairs
    phs_block = counts["_pair_lists"]["PHS"]
    phs_pairs = {
        tuple(sorted(p))
        for p in zip(ids[phs_block.first_rows].tolist(), ids[phs_block.second_rows].tolist(), strict=True)
    }
    assert (5, 7) not in phs_pairs
    assert phs_pairs == {(5, 6), (6, 7)}
