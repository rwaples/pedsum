"""The ``demography.multiple_births`` summary section (``summarize --multiple-birth-col``).

Counts the Multiple-Birth Sets of a validated pedigree, the twin pairs within
them by sex, and Weinberg's differential rule: with zygosity unknown,
opposite-sex twin pairs are DZ, DZ pairs are as often same-sex as
opposite-sex, so ``same - opposite`` estimates the MZ pairs and
``2 * opposite`` the DZ pairs.
"""

from __future__ import annotations

import polars as pl

from pedsum.base import SEX_FEMALE, SEX_MALE
from pedsum.redaction import null_below, redact_small_group

_SEX_MIX = ("male_male", "female_female", "male_female", "unknown_sex")


def compute_multiple_births(df: pl.DataFrame, pairs: dict) -> dict:
    """The section for ``df`` (a validated frame with ``multiple_birth``) and its relationship pair counts.

    A pair is two members of one set, so a set of ``k`` holds ``k(k-1)/2``.
    ``full_sib_pair_fraction`` is their share of the full-sib pairs (``FS``
    plus ``MZ``, since linked co-twins leave ``FS``); it is None when
    ``--max-degree`` left the full sibs uncounted. Weinberg's rule uses only
    sets of two whose sexes are both known. With ``birth_year``, a set's
    decade is its earliest known member's.
    """
    members = df.filter(pl.col("multiple_birth") >= 0)
    year = (
        pl.col("birth_year").filter(pl.col("birth_year") >= 0).min()
        if "birth_year" in df.columns
        else pl.lit(None, dtype=pl.Int32)
    )
    per_set = (
        members.group_by("multiple_birth")
        .agg(
            size=pl.len().cast(pl.Int64),
            n_male=(pl.col("sex") == SEX_MALE).sum().cast(pl.Int64),
            n_female=(pl.col("sex") == SEX_FEMALE).sum().cast(pl.Int64),
            decade=year // 10 * 10,
        )
        .with_columns(
            n=pl.col("size") * (pl.col("size") - 1) // 2,
            male_male=pl.col("n_male") * (pl.col("n_male") - 1) // 2,
            female_female=pl.col("n_female") * (pl.col("n_female") - 1) // 2,
            male_female=pl.col("n_male") * pl.col("n_female"),
        )
        .with_columns(unknown_sex=pl.col("n") - pl.col("male_male") - pl.col("female_female") - pl.col("male_female"))
    )
    n_pairs = int(per_set["n"].sum())
    full_sibs, mz = pairs.get("FS"), pairs.get("MZ", 0)
    full_sib_pairs = None if full_sibs is None or mz is None else full_sibs + mz
    sizes = per_set["size"].value_counts().sort("size")

    twins = per_set.filter((pl.col("size") == 2) & (pl.col("n_male") + pl.col("n_female") == 2))
    same = int((twins["male_female"] == 0).sum())
    opposite = len(twins) - same
    out: dict = {
        "n_sets": len(per_set),
        "n_individuals": len(members),
        "n_pairs": n_pairs,
        "sets_by_size": {str(s): int(c) for s, c in sizes.iter_rows()},
        "pairs_by_sex": {k: int(per_set[k].sum()) for k in _SEX_MIX},
        "full_sib_pair_fraction": n_pairs / full_sib_pairs if full_sib_pairs else None,
    }
    if "twin" in df.columns:
        out["mz_linked_pairs"] = int((df["twin"] >= 0).sum()) // 2
    out["weinberg"] = {
        "n_twin_pairs": len(twins),
        "same_sex": same,
        "opposite_sex": opposite,
        "mz_pairs_estimate": same - opposite,
        "dz_pairs_estimate": 2 * opposite,
        "mz_fraction_estimate": (same - opposite) / len(twins) if len(twins) else None,
    }
    if "birth_year" in df.columns:
        by_decade = per_set.group_by("decade").agg(pl.col("n", *_SEX_MIX).sum()).sort("decade", nulls_last=True)
        out["by_birth_decade"] = [
            {"decade": "unknown" if row["decade"] is None else int(row["decade"])}
            | {k: int(row[k]) for k in ("n", *_SEX_MIX)}
            for row in by_decade.iter_rows(named=True)
        ]
    return out


def redact_multiple_births(section: dict, min_cell: int, n_total: int) -> None:
    """``--safe-attempt``: null small counts, Weinberg's estimates when either of its counts is small, small decades."""
    null_below(section, ("n_sets", "n_individuals", "n_pairs", "mz_linked_pairs"), min_cell)
    null_below(section["sets_by_size"], list(section["sets_by_size"]), min_cell)
    null_below(section["pairs_by_sex"], _SEX_MIX, min_cell)
    if section["n_pairs"] is None:
        section["full_sib_pair_fraction"] = None
    weinberg = section["weinberg"]
    if any(0 < weinberg[k] < min_cell for k in ("same_sex", "opposite_sex")):
        for k in ("same_sex", "opposite_sex", "mz_pairs_estimate", "dz_pairs_estimate", "mz_fraction_estimate"):
            weinberg[k] = None
    null_below(weinberg, ("n_twin_pairs",), min_cell)
    for row in section.get("by_birth_decade", []):
        redact_small_group(row, ("decade",), min_cell, _SEX_MIX)
