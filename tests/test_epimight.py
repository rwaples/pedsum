"""Tests for the ``epimight-input`` subcommand and ``pedsum.epimight``.

Covers:

- the CLI emits ``pipeline_input.tsv`` with the EPIMIGHT schema, structural
  columns computed and phenotype columns left as empty placeholders;
- ``--parquet`` additionally writes the native parquet with nullable dtypes;
- exact per-relationship relative counts on a hand-built pedigree;
- ``born_at_year`` derivation (base-year + generation vs. real birth_year);
- ``--drop-founders`` and unknown-relationship-code handling.
"""

from __future__ import annotations

from collections import Counter

import polars as pl
from conftest import EXAMPLE, write_ped
from conftest import run_pedsum as _run

from pedsum.epimight import (
    EPIMIGHT_COLUMNS,
    EPIMIGHT_RELATIONSHIP_ORDER,
    PLACEHOLDER_COLUMNS,
    RELATIVE_PAIR_COLUMNS,
    build_epimight_skeleton,
    build_relative_pairs,
)

_DIRECTIONAL = {"PO", "Av", "1G"}

# --- a small deterministic pedigree with one of each relationship -----------
#
#   founders: 1(F) 2(M)  +  3(F)  +  4(M)
#   gen1:     5(F)=⟨1,2⟩  6(M)=⟨1,2⟩      → 5,6 full sibs
#   gen2:     7(M)=⟨5,4⟩  8(F)=⟨3,6⟩      → 7,8 first cousins (parents 5,6 sibs)
#
# Derived facts: 7's only avuncular is 6 (sib of parent 5); 8's is 5.
# 7's grandparents are 1,2 (parents of 5); 8's are 1,2 (parents of 6). No half sibs.
_PEDIGREE = [
    {"id": 1, "sex": "F", "mother": -1, "father": -1},
    {"id": 2, "sex": "M", "mother": -1, "father": -1},
    {"id": 3, "sex": "F", "mother": -1, "father": -1},
    {"id": 4, "sex": "M", "mother": -1, "father": -1},
    {"id": 5, "sex": "F", "mother": 1, "father": 2},
    {"id": 6, "sex": "M", "mother": 1, "father": 2},
    {"id": 7, "sex": "M", "mother": 5, "father": 4},
    {"id": 8, "sex": "F", "mother": 3, "father": 6},
]

# Inbred pedigree: full sibs 3,4 mate, so their children 5,6 are inbred full
# sibs. Exact kinship φ(5,6) = ¼[φ(3,3)+2φ(3,4)+φ(4,4)] = ¼[0.5+2(0.25)+0.5] =
# 0.375, vs the nominal FS coefficient 0.25.
_INBRED = [
    {"id": 1, "sex": "F", "mother": -1, "father": -1},
    {"id": 2, "sex": "M", "mother": -1, "father": -1},
    {"id": 3, "sex": "F", "mother": 1, "father": 2},
    {"id": 4, "sex": "M", "mother": 1, "father": 2},
    {"id": 5, "sex": "F", "mother": 3, "father": 4},
    {"id": 6, "sex": "M", "mother": 3, "father": 4},
]


def _load_graph(rows, tmp_path):
    """Write ``rows`` to a TSV, run pedsum's validator, and build the PedigreeGraph."""
    from pedsum.pairs import _build_pedigree_graph
    from pedsum.validate import load_and_validate

    ped = write_ped(tmp_path / "ped.tsv", rows)
    df, _ = load_and_validate(ped)
    return df, _build_pedigree_graph(df)


def _skeleton_from_rows(rows, tmp_path, **kwargs) -> pl.DataFrame:
    """Build the EPIMIGHT skeleton frame from raw pedigree rows."""
    df, pg = _load_graph(rows, tmp_path)
    return build_epimight_skeleton(df, pg, **kwargs)


def _pairs_from_rows(rows, tmp_path, **kwargs) -> pl.DataFrame:
    """Build the relative-pairs frame from raw pedigree rows."""
    df, pg = _load_graph(rows, tmp_path)
    return build_relative_pairs(df, pg, **kwargs)


def _relatives(frame: pl.DataFrame, person_id: str, rel: str) -> int:
    """Pull the ``relatives`` count for one (person, relationship_kind) cell."""
    block = frame.filter((pl.col("person_id").cast(pl.String) == person_id) & (pl.col("relationship_kind") == rel))
    return int(block["relatives"][0])


def _pair_set(pairs: pl.DataFrame, rel: str) -> set[tuple[int, int]]:
    """All ``(id1, id2)`` pairs for one relationship kind."""
    block = pairs.filter(pl.col("relationship_kind") == rel)
    return {(int(a), int(b)) for a, b in zip(block["id1"], block["id2"], strict=True)}


# --- CLI ---------------------------------------------------------------------


def test_cli_emits_tsv_skeleton(tmp_path):
    """`epimight-input` writes pipeline_input.tsv with the full EPIMIGHT schema."""
    out = tmp_path / "out"
    res = _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out)])
    assert res.returncode == 0, res.stderr

    tsv = out / "pipeline_input.tsv"
    assert tsv.exists()
    df = pl.read_csv(tsv, separator="\t")
    assert list(df.columns) == list(EPIMIGHT_COLUMNS)
    # 200 people × 8 relationship kinds.
    assert len(df) == 200 * len(EPIMIGHT_RELATIONSHIP_ORDER)
    assert sorted(df["relationship_kind"].unique().to_list()) == sorted(EPIMIGHT_RELATIONSHIP_ORDER)
    # Phenotype columns are unfillable from a pedigree → all empty.
    for col in PLACEHOLDER_COLUMNS:
        assert df[col].is_null().all(), col
    # Structural columns are populated.
    assert df["born_at_year"].is_not_null().all()
    assert df["relatives"].is_not_null().all()


def test_cli_parquet_opt_in(tmp_path):
    """`--parquet` also writes the native parquet with nullable integer dtypes."""
    out = tmp_path / "out"
    res = _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out), "--parquet"])
    assert res.returncode == 0, res.stderr

    assert (out / "pipeline_input.tsv").exists()
    parquet = out / "pipeline_input.parquet"
    assert parquet.exists()
    df = pl.read_parquet(parquet)
    # Placeholder columns keep schema-correct (nullable) integer dtype, all null.
    assert df.schema["failure_status"] == pl.Int8
    assert df.schema["relatives_diagnosed"] == pl.Int32
    assert df["failure_status"].is_null().all()
    assert df.schema["relatives"] == pl.Int32


def test_unknown_rel_code_exits_rc2(tmp_path):
    """An unknown relationship code is rejected before any work, rc=2."""
    out = tmp_path / "out"
    res = _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out), "--rels", "PO,XX"])
    assert res.returncode == 2, res.stderr
    assert "unknown relationship code" in res.stderr


def test_cli_disorder_and_rels_subset(tmp_path):
    """`--disorder` labels the single block and `--rels` restricts the kinds."""
    out = tmp_path / "out"
    res = _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out), "--rels", "PO,FS", "--disorder", "MDD"])
    assert res.returncode == 0, res.stderr
    df = pl.read_csv(out / "pipeline_input.tsv", separator="\t")
    assert df["disorder"].unique().to_list() == ["MDD"]
    assert sorted(df["relationship_kind"].unique().to_list()) == ["FS", "PO"]


# --- exact relationship counts ----------------------------------------------


def test_relationship_counts_exact(tmp_path):
    """Per-relationship relative counts match the hand-built pedigree."""
    frame = _skeleton_from_rows(_PEDIGREE, tmp_path)

    # Full sibs: 5 and 6 each have exactly one; founders/gen2 have none.
    assert _relatives(frame, "5", "FS") == 1
    assert _relatives(frame, "6", "FS") == 1
    assert _relatives(frame, "7", "FS") == 0

    # Parent-offspring (directional → counted on the child): non-founders have 2.
    assert _relatives(frame, "7", "PO") == 2
    assert _relatives(frame, "1", "PO") == 0

    # First cousins: 7 and 8 are each other's only cousin.
    assert _relatives(frame, "7", "1C") == 1
    assert _relatives(frame, "8", "1C") == 1

    # Avuncular (directional, oriented to the younger niece/nephew).
    assert _relatives(frame, "7", "Av") == 1
    assert _relatives(frame, "8", "Av") == 1
    assert _relatives(frame, "5", "Av") == 0  # aunt is not charged the relationship

    # Grandparent-grandchild (directional → counted on the grandchild): 1,2.
    assert _relatives(frame, "7", "1G") == 2
    assert _relatives(frame, "8", "1G") == 2

    # No half sibs anywhere in this pedigree.
    assert frame.filter(pl.col("relationship_kind").is_in(["HS", "mHS", "pHS"]))["relatives"].sum() == 0


def test_born_at_year_derived_from_generation(tmp_path):
    """Without a birth-year column, born_at_year = base_year + generation."""
    frame = _skeleton_from_rows(_PEDIGREE, tmp_path, base_year=2000)
    fs = frame.filter(pl.col("relationship_kind") == "FS")

    def _born(pid: str) -> int:
        return int(fs.filter(pl.col("person_id") == pid)["born_at_year"][0])

    # Founders (gen 0) → 2000; gen1 (5,6) → 2001; gen2 (7,8) → 2002.
    assert _born("1") == 2000
    assert _born("5") == 2001
    assert _born("7") == 2002


def test_drop_founders_removes_generation_zero(tmp_path):
    """`drop_founders` removes the founder generation; others are retained."""
    full = _skeleton_from_rows(_PEDIGREE, tmp_path)
    dropped = _skeleton_from_rows(_PEDIGREE, tmp_path, drop_founders=True)
    founders = {"1", "2", "3", "4"}
    assert founders.issubset(set(full["person_id"].cast(pl.String).to_list()))
    assert founders.isdisjoint(set(dropped["person_id"].cast(pl.String).to_list()))
    # Non-founders survive (e.g. the cousins).
    assert {"5", "6", "7", "8"}.issubset(set(dropped["person_id"].cast(pl.String).to_list()))


# --- relative pairs ----------------------------------------------------------


def test_pairs_only_under_flag(tmp_path):
    """relative_pairs.tsv is written only with --pairs; columns are id1,id2,kind,kinship."""
    out = tmp_path / "out"
    assert _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out)]).returncode == 0
    assert not (out / "relative_pairs.tsv").exists()

    out2 = tmp_path / "out2"
    assert _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out2), "--pairs"]).returncode == 0
    pairs = pl.read_csv(out2 / "relative_pairs.tsv", separator="\t")
    assert list(pairs.columns) == list(RELATIVE_PAIR_COLUMNS)
    assert len(pairs) > 0


def test_pairs_parquet_dtypes(tmp_path):
    """`--pairs --parquet` writes relative_pairs.parquet with a float kinship column."""
    out = tmp_path / "out"
    res = _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out), "--pairs", "--parquet"])
    assert res.returncode == 0, res.stderr
    pairs = pl.read_parquet(out / "relative_pairs.parquet")
    assert pairs.schema["kinship"] == pl.Float64
    assert set(pairs["relationship_kind"].to_list()) <= set(EPIMIGHT_RELATIONSHIP_ORDER)


def test_pairs_reconcile_with_skeleton_counts(tmp_path):
    """Pair counts back the skeleton: Σrelatives == 2·n_pairs (symmetric) or n_pairs (directional)."""
    skeleton = _skeleton_from_rows(_PEDIGREE, tmp_path)
    pairs = _pairs_from_rows(_PEDIGREE, tmp_path)
    for rel in EPIMIGHT_RELATIONSHIP_ORDER:
        n_pairs = int((pairs["relationship_kind"] == rel).sum())
        rel_sum = int(skeleton.filter(pl.col("relationship_kind") == rel)["relatives"].sum())
        expected = n_pairs if rel in _DIRECTIONAL else 2 * n_pairs
        assert rel_sum == expected, rel


def test_pairs_exact_and_orientation(tmp_path):
    """Exact pairs and orientation on the hand-built pedigree."""
    pairs = _pairs_from_rows(_PEDIGREE, tmp_path)

    # Symmetric kinds are canonicalized id1 < id2.
    assert _pair_set(pairs, "FS") == {(5, 6)}
    assert _pair_set(pairs, "1C") == {(7, 8)}
    sym = pairs.filter(~pl.col("relationship_kind").is_in(list(_DIRECTIONAL)))
    assert (sym["id1"] < sym["id2"]).all()

    # Directional kinds put the younger member first.
    assert _pair_set(pairs, "Av") == {(7, 6), (8, 5)}  # niece/nephew → aunt/uncle
    assert _pair_set(pairs, "1G") == {(7, 1), (7, 2), (8, 1), (8, 2)}  # grandchild → grandparent
    assert _pair_set(pairs, "PO") == {(5, 1), (5, 2), (6, 1), (6, 2), (7, 5), (7, 4), (8, 3), (8, 6)}

    # No half sibs in this pedigree.
    assert _pair_set(pairs, "HS") == set()

    # Kinship is the per-kind coefficient.
    by_kind = dict(pairs.group_by("relationship_kind").agg(pl.col("kinship").first()).iter_rows())
    assert by_kind["FS"] == 0.25
    assert by_kind["Av"] == 0.125
    assert by_kind["1C"] == 0.0625


def test_emitted_outputs_consistent_per_person(tmp_path):
    """The two emitted files reconcile per person.

    For every (person, kind), the skeleton's ``relatives`` equals how often that
    person appears in relative_pairs.tsv — as ``id1`` for directional kinds (only
    the younger member is charged) or on either side for symmetric kinds.
    """
    out = tmp_path / "out"
    res = _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out), "--pairs"])
    assert res.returncode == 0, res.stderr

    # Read ids as strings so the skeleton's person_id and the pairs' id1/id2 align.
    skeleton = pl.read_csv(out / "pipeline_input.tsv", separator="\t", schema_overrides={"person_id": pl.String})
    pairs = pl.read_csv(
        out / "relative_pairs.tsv", separator="\t", schema_overrides={"id1": pl.String, "id2": pl.String}
    )

    for rel in EPIMIGHT_RELATIONSHIP_ORDER:
        block_sk = skeleton.filter(pl.col("relationship_kind") == rel)
        block_pairs = pairs.filter(pl.col("relationship_kind") == rel)
        charged = (
            block_pairs["id1"].to_list()
            if rel in _DIRECTIONAL
            else [*block_pairs["id1"].to_list(), *block_pairs["id2"].to_list()]
        )
        from_pairs = Counter(charged)
        for person, relatives in zip(block_sk["person_id"], block_sk["relatives"], strict=True):
            assert from_pairs.get(person, 0) == relatives, (rel, person)


# --- exact (pedigree) kinship ------------------------------------------------


def test_exact_kinship_is_inbreeding_aware(tmp_path):
    """`kinship_exact` reflects the pedigree (inbreeding) while `kinship` stays nominal."""
    pairs = _pairs_from_rows(_INBRED, tmp_path, rels=("FS",), exact_kinship=True)
    assert "kinship_exact" in pairs.columns
    assert (pairs["kinship"] == 0.25).all()  # nominal is constant per kind

    inbred = pairs.filter((pl.col("id1") == 5) & (pl.col("id2") == 6)).row(0, named=True)
    assert inbred["kinship_exact"] == 0.375  # inbred full sibs (parents are sibs)
    outbred = pairs.filter((pl.col("id1") == 3) & (pl.col("id2") == 4)).row(0, named=True)
    assert outbred["kinship_exact"] == 0.25  # unrelated-founder parents


def test_cli_exact_kinship_adds_column(tmp_path):
    """`--pairs --exact-kinship` adds a kinship_exact column; without it, it is absent."""
    out = tmp_path / "out"
    res = _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out), "--pairs", "--exact-kinship"])
    assert res.returncode == 0, res.stderr
    with_exact = pl.read_csv(out / "relative_pairs.tsv", separator="\t")
    assert list(with_exact.columns) == [*RELATIVE_PAIR_COLUMNS, "kinship_exact"]

    out2 = tmp_path / "out2"
    assert _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out2), "--pairs"]).returncode == 0
    without = pl.read_csv(out2 / "relative_pairs.tsv", separator="\t")
    assert list(without.columns) == list(RELATIVE_PAIR_COLUMNS)


def test_exact_kinship_without_pairs_warns(tmp_path):
    """`--exact-kinship` without `--pairs` is a no-op that warns and writes no pairs file."""
    out = tmp_path / "out"
    res = _run(["epimight-input", "--in", str(EXAMPLE), "--out", str(out), "--exact-kinship"])
    assert res.returncode == 0, res.stderr
    assert "no effect without --pairs" in res.stderr
    assert not (out / "relative_pairs.tsv").exists()
