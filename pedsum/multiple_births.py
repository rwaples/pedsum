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

#: The sex mixes a pair of one set can have, in report order.
SEX_MIX = ("male_male", "female_female", "male_female", "unknown_sex")


def multiple_birth_sets(df: pl.DataFrame) -> pl.DataFrame:
    """One row per Multiple-Birth Set of ``df``: its size, members by sex, decade, and pairs by sex mix.

    A pair is two members of one set, so a set of ``k`` holds ``k(k-1)/2``.
    With ``birth_year``, a set's decade is its earliest known member's, else
    null.
    """
    members = df.filter(pl.col("multiple_birth") >= 0)
    year = (
        pl.col("birth_year").filter(pl.col("birth_year") >= 0).min()
        if "birth_year" in df.columns
        else pl.lit(None, dtype=pl.Int32)
    )
    return (
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


def twin_pairs(df: pl.DataFrame) -> pl.DataFrame:
    """The sets of two of :func:`multiple_birth_sets`, with the lower row of each as ``row`` and the other as ``co_twin``."""
    rows = (
        df.with_row_index("row")
        .filter(pl.col("multiple_birth") >= 0)
        .group_by("multiple_birth")
        .agg(row=pl.col("row").min().cast(pl.Int64), co_twin=pl.col("row").max().cast(pl.Int64))
    )
    return multiple_birth_sets(df).filter(pl.col("size") == 2).join(rows, on="multiple_birth")


def compute_multiple_births(df: pl.DataFrame, pairs: dict) -> dict:
    """The section for ``df`` (a validated frame with ``multiple_birth``) and its relationship pair counts.

    ``full_sib_pair_fraction`` is the share of the full-sib pairs (``FS``
    plus ``MZ``, since linked co-twins leave ``FS``) that are pairs of one
    set; it is None when ``--max-degree`` left the full sibs uncounted.
    Weinberg's rule uses only sets of two whose sexes are both known.
    """
    per_set = multiple_birth_sets(df)
    n_pairs = int(per_set["n"].sum())
    full_sibs, mz = pairs.get("FS"), pairs.get("MZ", 0)
    full_sib_pairs = None if full_sibs is None or mz is None else full_sibs + mz
    sizes = per_set["size"].value_counts().sort("size")

    twins = per_set.filter((pl.col("size") == 2) & (pl.col("n_male") + pl.col("n_female") == 2))
    same = int((twins["male_female"] == 0).sum())
    opposite = len(twins) - same
    out: dict = {
        "n_sets": len(per_set),
        "n_individuals": int(per_set["size"].sum()),
        "n_pairs": n_pairs,
        "sets_by_size": {str(s): int(c) for s, c in sizes.iter_rows()},
        "pairs_by_sex": {k: int(per_set[k].sum()) for k in SEX_MIX},
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
        by_decade = per_set.group_by("decade").agg(pl.col("n", *SEX_MIX).sum()).sort("decade", nulls_last=True)
        out["by_birth_decade"] = [
            {"decade": "unknown" if row["decade"] is None else int(row["decade"])}
            | {k: int(row[k]) for k in ("n", *SEX_MIX)}
            for row in by_decade.iter_rows(named=True)
        ]
    return out
