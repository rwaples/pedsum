"""``summarize --safe-attempt``: the release list, and the payload it is applied to.

Safe mode writes nothing the ordinary summary payload holds. It builds a
separate *source* payload of integer counts at full resolution
(:func:`build_source`), and :func:`pedsum.disclosure.build_release` copies
from it only the leaves :func:`summarize_spec` lists, each protected at the
minimum cell. :func:`safe_outputs` then arranges that release, and nothing
else, into the three output files.

What the list leaves out, and why:

* Every per-individual count of relatives (offspring, mates, sibs,
  ancestors, ...) and every sibship-size table. Their sums are relationship
  pair counts (``MO + FO`` is the total offspring count, ``FS`` the sum of
  ``k(k-1)/2`` over sibships), and a :class:`~pedsum.disclosure.Link` holds
  sums of cells, not weighted sums. A merged tail of such a table could be
  solved from the pair counts, so the pair counts are released and the
  tables are not.
* Every extremum, and every number that repeats another (a fraction beside
  its count, a complement, a rollup).
* Every concentration index over per-founder contributions: the effective
  number of founders, and the mean and sd of equivalent complete
  generations. Each narrows how the descendants are spread over the
  founders, and no count gate short of a dominance rule closes that.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from pedigree_graph import RELATIONSHIPS

from pedsum.base import INBRED_TOL, SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from pedsum.disclosure import (
    Contingency,
    Continuous,
    Count,
    Derived,
    Link,
    Plain,
    ReleaseSpec,
    Table,
    build_release,
    shared_total,
)
from pedsum.multiple_births import SEX_MIX, multiple_birth_sets
from pedsum.progress import relationship_progress
from pedsum.sections import F_HIST_EDGES, numeric_distribution
from pedsum.sex_concordance import GROUPINGS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pedigree_graph import PedigreeGraph, RelationshipCategory

    from pedsum.disclosure import CellRef, Path, Rule
    from pedsum.pedigree_ops import MatingPairs

_META = ("input", "command", "version", "generated_at")
_N: Path = ("n_total",)

_STRUCTURE: Path = ("pedigree", "structure")
_SIZE: Path = (*_STRUCTURE, "size_structure")
_SEX: Path = (*_SIZE, "sex")
_PARENTS: Path = (*_SIZE, "parents_known")
_DEPTH: Path = (*_SIZE, "depth")
_SEX_SOURCE: Path = (*_SIZE, "sex_source")
_CONCORDANCE: Path = ("pedigree", "demography", "offspring_sex_concordance")
#: Root of the gate tables: counts the release is tested against and linked to, written to no output.
_GATES = "gates"
_CONCORDANCE_GATES: Path = (_GATES, "offspring_sex_concordance")
#: The grouping every other grouping's eligible offspring include: full sibs share each parent.
_NARROWEST = GROUPINGS[0][0]
#: The parents-known classes a grouping's offspring come from, by the parent columns that define it.
_SCOPE: dict[tuple[str, ...], tuple[str, ...]] = {
    ("mother", "father"): ("both",),
    ("mother",): ("both", "mother_only"),
    ("father",): ("both", "father_only"),
}
_BIRTHS: Path = ("pedigree", "demography", "multiple_births")
_TWINS: Path = (*_BIRTHS, "twin_pairs_by_sex")
_TWIN_DECADES: Path = (*_BIRTHS, "twin_pairs_by_birth_decade")
_STATUS: Path = ("pedigree", "individuals", "reproduction", "by_status")
_PAIRS: Path = ("pedigree", "relatedness", "relationship_pairs")
_INBREEDING: Path = ("pedigree", "relatedness", "inbreeding")
_INBRED: Path = (*_INBREEDING, "inbred")
_F_HIST: Path = (*_INBREEDING, "F_hist")
_BY_SEX: Path = ("pedigree", "strata", "sex_summary")
_BY_DEPTH: Path = ("pedigree", "strata", "depth_summary")
_DEPTH_REPRODUCTIVE: Path = (*_BY_DEPTH, "reproductive")
_DEPTH_INBRED: Path = (*_BY_DEPTH, "inbred")
_DEPTH_PARENTS: Path = (*_BY_DEPTH, "parents_recorded")

_SEXES = (("female", SEX_FEMALE), ("male", SEX_MALE), ("unknown", SEX_UNKNOWN))
_PARENT_CODES = ("MO", "FO")
_RECORDED = ("both", "mother_only", "father_only")
_PARENT_CELLS = (*_RECORDED, "founder_with_descendants", "founder_without_descendants")

#: Names the pair counter of a safe run in its report.
PAIR_ENGINE = "rust_relatives_per_person"

# The codes that name the parent their pairs go through; every other code leaves the line open.
_SHARED_PARENT = {"MHS": "mother", "PHS": "father", "MO": "mother", "FO": "father"}

#: A statistic of a grouping that was not analysed. A string rather than an
#: absent key: absence would tell a withheld group count of 1 from one of 2
#: to 4.
NOT_COMPUTED = "not_computed"

# Without ``sensitivity_eligibility``: the all-resolved sensitivity analysis is not released.
_NULL_MODEL_KEYS = (
    "method",
    "statistic",
    "sex_probability",
    "min_group_size",
    "headline_eligibility",
    "multiplicity",
)


# ---------------------------------------------------------------------------
# Source payload
# ---------------------------------------------------------------------------


def family_ids(mother_rows: np.ndarray, father_rows: np.ndarray) -> np.ndarray:
    """Each row's family: rows with the same recorded parents share one, and a row with no parent is its own."""
    n = len(mother_rows)
    key = (mother_rows.astype(np.int64) + 1) * (n + 1) + (father_rows.astype(np.int64) + 1)
    parentless = np.flatnonzero(key == 0)
    key[parentless] = -1 - parentless
    return np.unique(key, return_inverse=True)[1]


@dataclass(frozen=True)
class PairCounts:
    """The pairs of each relationship code and the distinct units behind them (:func:`pair_counts`)."""

    counts: dict[str, int]
    support: dict[str, int]


def pair_counts(pg: PedigreeGraph, max_degree: int, mother_rows: np.ndarray, father_rows: np.ndarray) -> PairCounts:
    """Pairs and support of every relationship code up to ``max_degree``, from one pass over the relatives.

    ``PedigreeGraph.relatives_per_person`` credits a symmetric code to both
    members of a pair and a directional one (``GP``, ``Av``) to its junior
    member, so a code's pairs are its credits, halved when symmetric.
    Memory grows with rows times codes, not pairs.

    A code's support is the smaller of two counts, so pairs that all hang
    on one person or one couple have a support of 1 from either side:

    * the families (:func:`family_ids`) among the credited individuals;
    * the ancestors the pairs go through (:func:`_ancestor_units`).
    """
    with relationship_progress("relatives_per_person") as progress:
        relatives = pg.relatives_per_person(max_degree=max_degree, progress=progress)
    family = family_ids(mother_rows, father_rows)
    counts: dict[str, int] = {}
    support: dict[str, int] = {}
    for column, code in enumerate(relatives.categories):
        per_person = relatives.counts[:, column, 0]
        credited = np.flatnonzero(per_person)
        category = RELATIONSHIPS[code]
        n_credits = int(per_person.sum(dtype=np.int64))
        counts[code] = n_credits // 2 if category.symmetric else n_credits
        families = int(np.unique(family[credited]).size)
        ancestors = _ancestor_units(category, credited, mother_rows, father_rows)
        support[code] = families if ancestors is None else min(families, ancestors)
    return PairCounts(counts, support)


def _ancestor_units(
    category: RelationshipCategory, credited: np.ndarray, mother_rows: np.ndarray, father_rows: np.ndarray
) -> int | None:
    """The distinct ancestors the pairs of ``category`` go through, or a lower bound; None when none is recorded.

    Each credited individual is tied to its relatives of this code through
    an ancestor ``category.up`` generations above it. Which ancestor is
    known in two cases, and those are counted: one generation up it is the
    individual's own parent (the shared parent of half sibs, the mother
    and the father of full sibs), and in a lineal code (``GP``) it is every
    ancestor at that distance, the relative itself. Otherwise (``Av``,
    ``1C``) it could be any ancestor at that distance, and the bound is the
    fewest of them that could have every credited individual below them.

    A code that goes through a couple (``FS``, ``Av``, ``1C``, and ``MZ``
    through the twins' parents) takes the smaller of the count over mothers
    and the count over fathers, so one woman's children by five men are
    one unit. A lineal count is over by the few ancestors that inbreeding
    also places closer to the same individual.
    """
    up = max(category.up, 1)
    reached = np.zeros(len(mother_rows), dtype=np.int64)
    reached[credited] = 1
    for _ in range(up - 1):
        reached = _lift(reached, mother_rows) + _lift(reached, father_rows)
    by_line = {"mother": _lift(reached, mother_rows), "father": _lift(reached, father_rows)}
    one_line = category.code in _SHARED_PARENT or category.ancestor_count == 2 or category.code == "MZ"
    if category.code in _SHARED_PARENT:
        loads = [by_line[_SHARED_PARENT[category.code]]]
    elif one_line:
        loads = list(by_line.values())
    else:
        loads = [by_line["mother"] + by_line["father"]]
    known = (up == 1 and one_line) or category.down == 0
    bounds = [int(np.count_nonzero(load)) if known else _fewest_covering(load, len(credited)) for load in loads]
    return min((bound for bound in bounds if bound), default=None)


def _fewest_covering(load: np.ndarray, n: int) -> int:
    """The fewest rows whose ``load`` sums to ``n``, taking the largest first; 0 when no row has any."""
    largest_first = np.sort(load[load > 0])[::-1]
    return min(int(np.searchsorted(np.cumsum(largest_first), n)) + 1, len(largest_first))


def _lift(reached: np.ndarray, parent_rows: np.ndarray) -> np.ndarray:
    """Per row, the sum of ``reached`` over the rows that name it as this parent."""
    recorded = parent_rows >= 0
    return np.bincount(parent_rows[recorded], weights=reached[recorded], minlength=len(parent_rows)).astype(np.int64)


def _tally(codes: np.ndarray, labels: Mapping[str, int]) -> dict[str, int]:
    return {label: int((codes == code).sum()) for label, code in labels.items()}


def _split(mask: np.ndarray, yes: str, no: str) -> dict[str, int]:
    n_yes = int(mask.sum())
    return {yes: n_yes, no: len(mask) - n_yes}


def _cross(rows: Mapping[str, np.ndarray], cols: Mapping[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """Counts of individuals in each row mask and column mask."""
    return {r: {c: int((row & col).sum()) for c, col in cols.items()} for r, row in rows.items()}


def _yes_no(mask: np.ndarray, yes: str, no: str) -> dict[str, np.ndarray]:
    return {yes: mask, no: ~mask}


def _mean_by(rows: Mapping[str, np.ndarray], values: np.ndarray) -> dict[str, float]:
    return {r: float(values[mask].mean()) for r, mask in rows.items()}


def build_source(
    *,
    meta: Mapping[str, Any],
    max_degree: int,
    df: pl.DataFrame,
    idf: pl.DataFrame,
    mating: MatingPairs,
    n_components: int,
    sex_source: Mapping[str, int] | None,
    pairs: PairCounts,
    sex_concordance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The payload the release list is applied to: integer counts at full resolution, and the statistics over them.

    Built from the tables ``summarize`` already holds, not from the report
    payload, whose histograms are shares and top-coded. Holds more than is
    released (each pair count's support); only :func:`summarize_spec`'s
    leaves leave it.

    Which keys it holds depends on the options and the input's columns
    only. A statistic the data gives no value for is ``None`` under its
    key, which the release writes as it writes a withheld one.
    """
    n = len(idf)
    sex = idf["sex"].to_numpy()
    depth = idf["ped_depth"].to_numpy()
    has_mother = idf["mother"].to_numpy() != -1
    has_father = idf["father"].to_numpy() != -1
    founder = idf["is_founder"].to_numpy().astype(bool)
    reproductive = idf["n_offspring"].to_numpy() > 0
    descendant_paths = idf["n_descendant_paths"].to_numpy()

    by_sex = {label: sex == code for label, code in _SEXES if (sex == code).any()}
    by_depth = {str(d): depth == d for d in range(int(depth.max(initial=-1)) + 1)}
    status_cols = _yes_no(reproductive, "reproductive", "terminal")

    size_structure: dict[str, Any] = {
        "sex": _tally(sex, dict(_SEXES)),
        "parents_known": {
            "both": int((has_mother & has_father).sum()),
            "mother_only": int((has_mother & ~has_father).sum()),
            "father_only": int((~has_mother & has_father).sum()),
            "founder_with_descendants": int((founder & reproductive).sum()),
            "founder_without_descendants": int((founder & ~reproductive).sum()),
        },
        "depth": {label: int(mask.sum()) for label, mask in by_depth.items()},
        "depth_stats": numeric_distribution(depth),
    }
    if sex_source is not None:
        size_structure["sex_source"] = dict(sex_source)
    sex_summary: dict[str, Any] = {"reproductive": _cross(by_sex, status_cols)}
    recorded_cols = {
        "both": has_mother & has_father,
        "mother_only": has_mother & ~has_father,
        "father_only": ~has_mother & has_father,
        "none": ~has_mother & ~has_father,
    }
    depth_summary: dict[str, Any] = {
        "sex": _cross(by_depth, by_sex),
        "reproductive": _cross(by_depth, status_cols),
        "parents_recorded": _cross(by_depth, recorded_cols),
        "mean_descendant_paths": _mean_by(by_depth, descendant_paths),
    }
    if "n_distinct_ancestors" in idf.columns:
        depth_summary["mean_distinct_ancestors"] = _mean_by(by_depth, idf["n_distinct_ancestors"].to_numpy())

    relatedness: dict[str, Any] = {}
    if "F" in idf.columns:
        F = idf["F"].to_numpy()
        inbred = F > INBRED_TOL
        inbred_cols = _yes_no(inbred, "inbred", "not_inbred")
        F_inbred = F[inbred]
        relatedness["inbreeding"] = {
            "inbred": _split(inbred, "inbred", "not_inbred"),
            "F_hist": {"0": int((~inbred).sum())}
            | {f"<{hi:g}": int(((lo < F) & (hi >= F)).sum()) for lo, hi in pairwise(F_HIST_EDGES)},
            "mean_F_inbred": float(F_inbred.mean()) if len(F_inbred) else None,
            "sd_F_inbred": float(F_inbred.std(ddof=1)) if len(F_inbred) > 1 else None,
        }
        sex_summary["inbred"] = _cross(by_sex, inbred_cols)
        depth_summary["inbred"] = _cross(by_depth, inbred_cols)
        depth_summary["mean_F"] = _mean_by(by_depth, F)

    # A parent-offspring pair is one recorded parent link, which the parents-known table already counts.
    codes = [code for code in RELATIONSHIPS if code not in _PARENT_CODES and code in pairs.counts]
    relatedness["relationship_pairs"] = {
        "engine": PAIR_ENGINE,
        "by_code": {code: pairs.counts[code] for code in codes},
        "support": {code: pairs.support[code] for code in codes},
    }

    demography: dict[str, Any] = {"mating_pairs": {"n_pairs": len(mating.sizes)}}
    gates: dict[str, Any] = {}
    if sex_concordance is not None:
        concordance = _concordance_source(sex_concordance)
        demography["offspring_sex_concordance"] = concordance
        gates["offspring_sex_concordance"] = _concordance_gates(concordance, size_structure["parents_known"])
    if "multiple_birth" in df.columns:
        demography["multiple_births"] = _multiple_births_source(df)

    return {
        **{key: meta[key] for key in _META},
        "n_total": n,
        _GATES: gates,
        "pedigree": {
            "structure": {
                "max_degree_enumerated": max_degree,
                "size_structure": size_structure,
                "components": {"n_components": int(n_components)},
            },
            "demography": demography,
            "individuals": {"reproduction": {"by_status": _split(reproductive, "reproductive", "terminal")}},
            "relatedness": relatedness,
            "strata": {"sex_summary": sex_summary, "depth_summary": depth_summary},
        },
    }


_TEST_KEYS = (
    "observed_pair_concordance",
    "excess_concordance",
    "direction",
    "z",
    "p_analytical",
    "p_analytical_holm",
    "p_source",
    "p_holm",
)


def _concordance_source(section: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"null_model": {key: section["null_model"][key] for key in _NULL_MODEL_KEYS}}
    for name, _ in GROUPINGS:
        block = section[name]
        n_eligible = block["n_offspring_eligible"]
        # The fraction is n_male / n_eligible, so the product is that integer to within float rounding.
        n_male = round((block["conditioning_male_fraction"] or 0.0) * n_eligible)
        test: Any = NOT_COMPUTED
        if block["computed"]:
            proportion = block["male_proportion_distribution"]
            permutations = block["permutations"]
            test = {key: block[key] for key in _TEST_KEYS} | {
                "male_proportion_mean": proportion.mean,
                "male_proportion_sd": proportion.std,
                "permutations": None
                if permutations is None
                else {key: permutations[key] for key in ("requested", "completed", "seed", "p_raw", "p_holm")},
            }
        out[name] = {
            "n_groups_eligible": block["n_groups_eligible"],
            "eligible_by_sex": {"male": n_male, "female": n_eligible - n_male},
            "test": test,
        }
    return out


def _concordance_gates(concordance: Mapping[str, Any], parents_known: Mapping[str, int]) -> dict[str, Any]:
    """What each grouping's eligible counts differ by from the counts they are nested with, and the support that gives.

    A grouping's eligible offspring are among the individuals with its
    parent recorded, and include the eligible offspring of the full
    sibships. Each difference is a count a reader gets by subtraction, so
    the smallest nonzero one is the support of the grouping's counts.
    """
    narrow = concordance[_NARROWEST]
    out: dict[str, Any] = {}
    for name, parent_cols in GROUPINGS:
        block = concordance[name]
        by_sex = block["eligible_by_sex"]
        gate: dict[str, Any] = {
            "not_eligible": sum(parents_known[cell] for cell in _SCOPE[parent_cols]) - sum(by_sex.values())
        }
        differences = [gate["not_eligible"]]
        if name != _NARROWEST:
            gate["beyond_sibship_by_sex"] = {sex: by_sex[sex] - narrow["eligible_by_sex"][sex] for sex in by_sex}
            differences += gate["beyond_sibship_by_sex"].values()
            differences.append(abs(block["n_groups_eligible"] - narrow["n_groups_eligible"]))
        smallest = min((d for d in differences if d > 0), default=None)
        gate["groups_support"] = block["n_groups_eligible"] if smallest is None else smallest
        gate["by_sex_support"] = {sex: count if smallest is None else smallest for sex, count in by_sex.items()}
        out[name] = gate
    return out


def _multiple_births_source(df: pl.DataFrame) -> dict[str, Any]:
    per_set = multiple_birth_sets(df)
    twins = per_set.filter(pl.col("size") == 2)
    by_sex = {mix: int(twins[mix].sum()) for mix in SEX_MIX}
    same, opposite = by_sex["male_male"] + by_sex["female_female"], by_sex["male_female"]
    out: dict[str, Any] = {
        "twin_pairs_by_sex": by_sex,
        "weinberg": {"mz_fraction_estimate": (same - opposite) / (same + opposite) if same + opposite else None},
    }
    if "birth_year" in df.columns:
        decades = twins.group_by("decade").agg(pl.col(*SEX_MIX).sum()).sort("decade", nulls_last=True)
        out["twin_pairs_by_birth_decade"] = {
            "unknown" if row["decade"] is None else str(row["decade"]): {mix: int(row[mix]) for mix in SEX_MIX}
            for row in decades.iter_rows(named=True)
        } or None
    return out


# ---------------------------------------------------------------------------
# Release list
# ---------------------------------------------------------------------------

_PERSONS = "Individuals in one class of a split of the whole pedigree"
_GATE = "at the minimum cell, or the whole table is withheld"


def _rules() -> tuple[Rule, ...]:
    rules: list[Rule] = [
        Plain((key,), "Describes the run, not the individuals: the command, its input path, version and time.")
        for key in _META
    ]
    rules += [
        Count(_N, "The number of individuals in the pedigree, withheld when below the minimum cell."),
        Plain(
            (*_STRUCTURE, "max_degree_enumerated"),
            "The --max-degree option of the run, which the data does not set.",
        ),
        Table(_SEX, f"{_PERSONS} by sex, every class {_GATE}.", ordered=False),
        Table(
            _PARENTS,
            f"{_PERSONS} by which parents are recorded, founders by whether they have descendants, every class "
            f"{_GATE}; founder, link and edge counts are sums of these five and are not repeated.",
            ordered=False,
        ),
        Table(
            _DEPTH,
            f"{_PERSONS} by depth; a small deepest or shallowest depth is merged into its neighbour until the "
            "merged bin reaches the minimum cell.",
            ordered=True,
        ),
        Continuous(
            (*_SIZE, "depth_stats"),
            "Mean, sd and quartiles of depth, released only when the depth table needed no merging, so they "
            "cannot be used to split a merged bin.",
            histogram=_DEPTH,
        ),
        Table(_SEX_SOURCE, f"{_PERSONS} by where the sex came from, every class {_GATE}.", ordered=False),
        Count(
            (*_STRUCTURE, "components", "n_components"),
            "The number of connected components, withheld when 1 to 4; no component's size is released.",
        ),
        Count(
            ("pedigree", "demography", "mating_pairs", "n_pairs"),
            "The number of mating pairs, each a distinct couple, withheld when below the minimum cell; "
            "the sizes of their sibships are not released.",
        ),
        Table(
            _STATUS,
            f"{_PERSONS} by whether they have recorded offspring, both classes {_GATE}; "
            "the complement is a cell of the table, not a second number. It is the column totals of the "
            "depth-by-offspring table, whose deepest row is known, so it is released only with that table.",
            ordered=False,
            requires=(_DEPTH_REPRODUCTIVE,),
        ),
        Plain((*_PAIRS, "engine"), "Names the counting engine, a property of the software."),
        Table(
            (*_PAIRS, "by_code"),
            "Pairs by closest relationship. A code is released only when its pairs rest on at least the "
            "minimum cell of distinct families and of distinct ancestors they go through, and one small code "
            "withholds the table. MO and FO are left out because they are sums of parents-known cells; PO "
            "and the degree rollups are sums of codes.",
            ordered=False,
            support=(*_PAIRS, "support"),
        ),
        Table(
            _INBRED,
            f"{_PERSONS} by whether F exceeds zero, both classes {_GATE}. It is the column totals of the "
            "depth-by-inbred table, whose first two rows are known, so it is released only with that table.",
            ordered=False,
            requires=(_DEPTH_INBRED,),
        ),
        Table(
            _F_HIST,
            f"{_PERSONS} by F bucket, zero first; a small top or bottom bucket is merged into its neighbour, "
            "and the buckets are ranges, so no F value is released. Its zero bucket is the count of the "
            "individuals who are not inbred, so it is released on the same condition as that count.",
            ordered=True,
            requires=(_DEPTH_INBRED,),
        ),
        Derived(
            (*_INBREEDING, "mean_F_inbred"),
            "Mean F over the inbred individuals, released only when at least the minimum cell are inbred and "
            "the F buckets needed no merging: the mean would split a merged bucket, and a mean over all "
            "individuals would be the sum of a few values when few are inbred.",
            sources=(_INBRED, _F_HIST),
        ),
        Derived(
            (*_INBREEDING, "sd_F_inbred"),
            "Sd of F over the inbred individuals, on the same condition as their mean.",
            sources=(_INBRED, _F_HIST),
        ),
    ]
    for name, what in (("reproductive", "whether they have offspring"), ("inbred", "whether F exceeds zero")):
        rules.append(
            Contingency(
                (*_BY_SEX, name),
                f"Individuals by sex and {what}; released whole, only when no cell is small, and only with "
                "the table by depth, whose column totals it repeats.",
                requires=((*_BY_DEPTH, name),),
            )
        )
    for name, what in (
        ("sex", "sex"),
        ("reproductive", "whether they have offspring"),
        ("inbred", "whether F exceeds zero"),
        ("parents_recorded", "which parents are recorded"),
    ):
        rules.append(
            Contingency(
                (*_BY_DEPTH, name),
                f"Individuals by depth and {what}; released whole, and only when no cell is small.",
            )
        )
    rules += [
        Derived(
            (*_BY_DEPTH, "mean_F"),
            "Mean F per depth, released whole and only when at every depth the inbred are none or at least "
            "the minimum cell, and only with the inbred count and unmerged F buckets: a depth's mean times "
            "its size is its F sum, which over the only inbred depth is the withheld mean F of the inbred "
            "and splits a merged bucket.",
            sources=(_DEPTH_INBRED, _INBRED, _F_HIST),
        ),
        Derived(
            (*_BY_DEPTH, "mean_distinct_ancestors"),
            "Mean ancestor count per depth, released whole and only when the depth, parents-known and "
            "depth-by-recorded-parents tables are intact: at depth 1 it is the mean number of recorded "
            "parents, which with the depth's size gives the count of individuals with one.",
            sources=(_DEPTH, _PARENTS, _DEPTH_PARENTS),
        ),
        Derived(
            (*_BY_DEPTH, "mean_descendant_paths"),
            "Mean descendant-path count per depth, on the same condition as the ancestor means and only when "
            "the depth-by-offspring table is intact: over the parents of the deepest depth it is the mean "
            "number of recorded children, a count of parent links.",
            sources=(_DEPTH, _PARENTS, _DEPTH_PARENTS, _DEPTH_REPRODUCTIVE),
        ),
    ]
    rules += [
        Plain((*_CONCORDANCE, "null_model", key), "A fixed description of the test, the same for every pedigree.")
        for key in _NULL_MODEL_KEYS
    ]
    for name, _ in GROUPINGS:
        groups: Path = (*_CONCORDANCE, name, "n_groups_eligible")
        by_sex: Path = (*_CONCORDANCE, name, "eligible_by_sex")
        gate: Path = (*_CONCORDANCE_GATES, name)
        rules += [
            Count(
                groups,
                "Offspring groups the test uses, each a distinct family, withheld when 1 to 4, and when the "
                "grouping differs from the full sibships or from the individuals with its parent recorded by "
                "1 to 4 groups or offspring of a sex: the groupings are nested, so that difference is a count.",
                support=(*gate, "groups_support"),
            ),
            Table(
                by_sex,
                f"Offspring the test uses, by sex, both classes {_GATE}, withheld with its total on the same "
                "condition as the group count.",
                ordered=False,
                support=(*gate, "by_sex_support"),
            ),
            Count(
                (*gate, "not_eligible"),
                "Gate, never written: the individuals with the grouping's parent recorded whom the test does not use.",
            ),
        ]
        if name != _NARROWEST:
            rules.append(
                Table(
                    (*gate, "beyond_sibship_by_sex"),
                    "Gate, never written: the offspring the grouping uses beyond those of the full sibships, by sex.",
                    ordered=False,
                )
            )
        rules += [
            Derived(
                (*_CONCORDANCE, name, "test"),
                "The test's statistics over all eligible groups, released only when the groups and both "
                "sexes of their offspring reach the minimum cell; no per-group or per-size row is released.",
                sources=(groups, by_sex),
            ),
        ]
    rules += [
        Table(
            _TWINS,
            f"Multiple-birth sets of two, each a distinct family, by the sexes of the pair, every class {_GATE}; "
            "Weinberg's counts are sums of these cells, and larger sets are not released.",
            ordered=False,
        ),
        Derived(
            (*_BIRTHS, "weinberg", "mz_fraction_estimate"),
            "Weinberg's MZ share, a function of the twin-pair table, released only when that table is intact.",
            sources=(_TWINS,),
        ),
        Contingency(
            _TWIN_DECADES,
            "Sets of two by birth decade and sexes of the pair; released whole, and only when no cell is small.",
        ),
    ]
    return tuple(rules)


_RULES = _rules()

_ALL = "Every individual is counted once in each of these tables."


def _margins(
    table: Path,
    cells: Mapping[str, Mapping[str, int]],
    rows: Path | None,
    cols: Mapping[str, CellRef],
) -> list[Link]:
    """Links tying a contingency table's row sums to the cells of ``rows`` and its column sums to ``cols``."""
    row_labels = list(cells)
    col_labels = list(next(iter(cells.values()), {}))
    links = []
    if rows is not None:
        links += [
            Link(
                (rows, r), tuple((table, (r, c)) for c in col_labels), "A row of the table is one class of its margin."
            )
            for r in row_labels
        ]
    links += [
        Link(whole, tuple((table, (r, c)) for r in row_labels), "A column of the table is one class of its margin.")
        for c, whole in cols.items()
        if c in col_labels
    ]
    return links


def summarize_spec(source: Mapping[str, Any]) -> ReleaseSpec:
    """The release list for ``summarize``, with the links among the tables ``source`` holds.

    The rules are fixed. The links name table cells, and the rows of the
    by-sex, by-depth and by-decade tables are read from ``source``.
    """
    strata = source["pedigree"]["strata"]
    sex_cells = {label: (_SEX, label) for label, _ in _SEXES}
    status_cells = {label: (_STATUS, label) for label in ("reproductive", "terminal")}
    inbred_cells = {label: (_INBRED, label) for label in ("inbred", "not_inbred")}
    links = [
        *shared_total(
            _ALL,
            _N,
            _SEX,
            _PARENTS,
            _DEPTH,
            _SEX_SOURCE,
            _STATUS,
            _INBRED,
            _F_HIST,
        ),
        Link((_INBRED, "not_inbred"), ((_F_HIST, "0"),), "The zero bucket is the individuals who are not inbred."),
    ]
    founder_cells = {"reproductive": "founder_with_descendants", "terminal": "founder_without_descendants"}
    if "0" in source["pedigree"]["structure"]["size_structure"]["depth"]:
        links.append(
            Link(
                (_DEPTH, "0"),
                tuple((_PARENTS, cell) for cell in founder_cells.values()),
                "Depth 0 is the individuals with no parent.",
            )
        )
    for name, cols in (("reproductive", status_cells), ("inbred", inbred_cells)):
        if name in strata["sex_summary"]:
            links += _margins((*_BY_SEX, name), strata["sex_summary"][name], _SEX, cols)
    for name, cols in (("sex", sex_cells), ("reproductive", status_cells), ("inbred", inbred_cells)):
        if name in strata["depth_summary"]:
            links += _margins((*_BY_DEPTH, name), strata["depth_summary"][name], _DEPTH, cols)
    by_depth = strata["depth_summary"]
    depths = list(by_depth["reproductive"])
    if depths:
        links += [
            Link(
                (_PARENTS, cell),
                ((_DEPTH_REPRODUCTIVE, ("0", status)),),
                "A founder has descendants exactly when it has offspring, and the founders are depth 0.",
            )
            for status, cell in founder_cells.items()
        ]
        links.append(
            Link(
                (_DEPTH_REPRODUCTIVE, (depths[-1], "reproductive")),
                (),
                "An individual's offspring are deeper than it is, so the deepest depth has none.",
            )
        )
        links += _margins(_DEPTH_PARENTS, by_depth["parents_recorded"], _DEPTH, {c: (_PARENTS, c) for c in _RECORDED})
        links.append(
            Link(
                (_DEPTH, "0"),
                tuple((_DEPTH_PARENTS, (depth, "none")) for depth in depths),
                "Depth 0 is the individuals with no parent.",
            )
        )
        links += [
            Link((_DEPTH_PARENTS, (depth, "none")), (), "Below depth 0 every individual has a recorded parent.")
            for depth in depths[1:]
        ]
        links += [
            Link((_DEPTH_PARENTS, ("0", col)), (), "No individual of depth 0 has a recorded parent.")
            for col in _RECORDED
        ]
    if "inbred" in by_depth:
        links += [
            Link(
                (_DEPTH_INBRED, (depth, "inbred")),
                (),
                "F is zero for a founder and for the offspring of founders, who are unrelated.",
            )
            for depth in depths[:2]
        ]
    if "offspring_sex_concordance" in source[_GATES]:
        narrow: Path = (*_CONCORDANCE, _NARROWEST, "eligible_by_sex")
        for name, parent_cols in GROUPINGS:
            by_sex: Path = (*_CONCORDANCE, name, "eligible_by_sex")
            gate: Path = (*_CONCORDANCE_GATES, name)
            outside = tuple((_PARENTS, cell) for cell in _PARENT_CELLS if cell not in _SCOPE[parent_cols])
            links.append(
                Link(
                    (_PARENTS, None),
                    ((by_sex, None), ((*gate, "not_eligible"), None), *outside),
                    "The offspring a grouping uses are among the individuals with its parent recorded.",
                )
            )
            if name != _NARROWEST:
                links += [
                    Link(
                        (by_sex, sex),
                        ((narrow, sex), ((*gate, "beyond_sibship_by_sex"), sex)),
                        "Full sibs share each parent, so an eligible sibship's offspring are eligible in both "
                        "single-parent groupings.",
                    )
                    for sex in ("male", "female")
                ]
    decades = source["pedigree"]["demography"].get("multiple_births", {}).get("twin_pairs_by_birth_decade")
    if decades:
        links += _margins(_TWIN_DECADES, decades, None, {mix: (_TWINS, mix) for mix in SEX_MIX})
    return ReleaseSpec(_RULES, tuple(links))


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

#: Released leaves under these paths go to ``summary.extra.yaml``; the rest to ``summary.yaml``.
_EXTRA: tuple[Path, ...] = (
    _SEX_SOURCE,
    _F_HIST,
    (*_INBREEDING, "sd_F_inbred"),
    _BY_DEPTH,
    _TWIN_DECADES,
    *((*_CONCORDANCE, name, "eligible_by_sex") for name, _ in GROUPINGS),
)


@dataclass(frozen=True)
class SafeOutputs:
    """The three safe output payloads, each holding released leaves only."""

    slim: dict[str, Any]
    extra: dict[str, Any]
    pedigree_tsv: dict[str, Any]


def safe_outputs(source: Mapping[str, Any], min_cell: int) -> SafeOutputs:
    """Release ``source`` at ``min_cell`` and lay the release out as the two summary YAMLs and the pedigree long TSV.

    Every value in the result is read from the release at a rule's path.
    The YAMLs split the leaves between them and both start with the meta
    block, which records ``min_cell``. The TSV payload is flat, with a section per summary section.
    There is no individual one, since no per-individual statistic is released.
    """
    release = build_release(source, summarize_spec(source), min_cell)
    meta = {key: release[key] for key in (*_META, *_N) if key in release} | {"safe_attempt_min_cell": min_cell}
    slim: dict[str, Any] = dict(meta)
    extra: dict[str, Any] = dict(meta)
    for rule in _RULES:
        if len(rule.path) == 1 or rule.path[0] == _GATES:
            continue
        node: Any = release
        for key in rule.path:
            node = node.get(key, _MISSING) if isinstance(node, dict) else _MISSING
            if node is _MISSING:
                break
        if node is _MISSING:
            continue
        target = extra if any(rule.path[: len(prefix)] == prefix for prefix in _EXTRA) else slim
        for key in rule.path[:-1]:
            target = target.setdefault(key, {})
        target[rule.path[-1]] = node
    sections = {name: value for category in release.get("pedigree", {}).values() for name, value in category.items()}
    return SafeOutputs(slim, extra, {**meta, **sections})


_MISSING = object()
