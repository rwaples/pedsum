"""Per-section summary computations over the pedigree / individual table."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import polars as pl
import scipy.sparse.csgraph as csgraph
from pedigree_graph import RELATIONSHIPS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pedigree_graph import PedigreeGraph, RelationshipBurden

    from pedsum.pedigree_ops import MatingPairs

from pedsum.base import INBRED_TOL, SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from pedsum.pedigree_ops import _build_children_csr


def compute_size_structure(
    df: pl.DataFrame,
    mother_rows: np.ndarray,
    father_rows: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Counts, sex breakdown, generation depth, connected components.

    ``mother_rows`` / ``father_rows`` give each row's parent row, ``-1`` when
    unknown. Returns (summary_dict, component_labels) where
    component_labels[i] is the connected-component id for row i.
    """
    n = len(df)
    has_mom = mother_rows >= 0
    has_dad = father_rows >= 0
    has_parent = has_mom | has_dad
    has_both_parents = has_mom & has_dad
    n_founders = int((~has_parent).sum())
    n_nonfounders = int(has_parent.sum())
    n_mother_links = int(has_mom.sum())
    n_father_links = int(has_dad.sum())

    sex = df["sex"].to_numpy()
    n_male = int((sex == SEX_MALE).sum())
    n_female = int((sex == SEX_FEMALE).sum())
    n_unknown = int((sex == SEX_UNKNOWN).sum())

    gen = df["ped_depth"].to_numpy()
    max_depth = int(gen.max()) if n > 0 else 0
    gen_counts = np.bincount(gen).tolist() if n > 0 else []

    children_csr = _build_children_csr(mother_rows, father_rows)
    if children_csr is not None:
        n_components, comp_labels = csgraph.connected_components(children_csr, directed=False)
    else:
        n_components = n
        comp_labels = np.arange(n, dtype=np.int32)

    comp_sizes = np.bincount(comp_labels) if n > 0 else np.array([], dtype=np.int64)
    sorted_sizes = np.sort(comp_sizes)[::-1]

    summary = {
        "n_total": n,
        "n_founders": n_founders,
        "founder_frac": n_founders / n if n else 0.0,
        "n_nonfounders": n_nonfounders,
        "nonfounder_frac": n_nonfounders / n if n else 0.0,
        "n_male": n_male,
        "n_female": n_female,
        "n_unknown_sex": n_unknown,
        "n_mother_links": n_mother_links,
        "n_father_links": n_father_links,
        "n_parent_child_edges": n_mother_links + n_father_links,
        "n_with_both_parents": int(has_both_parents.sum()),
        "n_with_mother_only": int((has_mom & ~has_dad).sum()),
        "n_with_father_only": int((~has_mom & has_dad).sum()),
        "n_half_founders": int((has_mom ^ has_dad).sum()),
        "max_depth": max_depth,
        "mean_depth": float(gen.mean()) if n else 0.0,
        "median_depth": float(np.median(gen)) if n else 0.0,
        "depth_counts": gen_counts,
        "n_components": int(n_components),
        "largest_component": int(sorted_sizes[0]) if len(sorted_sizes) else 0,
        "largest_component_frac": (int(sorted_sizes[0]) / n) if len(sorted_sizes) and n else 0.0,
        "next_components": sorted_sizes[1:6].tolist(),
    }
    return summary, comp_labels


def _numeric_distribution(values: pl.Series | np.ndarray) -> dict:
    """Compact distribution summary for pedigree-level aggregate sections."""
    arr = values.to_numpy() if isinstance(values, pl.Series) else np.asarray(values)
    n = len(arr)
    is_float = np.issubdtype(arr.dtype, np.floating)
    cast = float if is_float else int
    return {
        "mean": float(arr.mean()) if n else 0.0,
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": cast(arr.min()) if n else 0,
        "q1": cast(np.quantile(arr, 0.25)) if n else 0,
        "median": cast(np.median(arr)) if n else 0,
        "q3": cast(np.quantile(arr, 0.75)) if n else 0,
        "max": cast(arr.max()) if n else 0,
        "nz": int((arr != 0).sum()) if n else 0,
    }


def _effective_count_from_weights(weights: np.ndarray) -> float:
    """Return 1 / sum(p_i^2), or 0 when no positive weights are present."""
    positive = weights[weights > 0].astype(np.float64)
    total = float(positive.sum())
    if total == 0.0:
        return 0.0
    p = positive / total
    return float(1.0 / np.sum(p * p))


def compute_mating_pair_summary(pair_sizes: np.ndarray) -> dict | None:
    """Aggregate per-Mating-Pair statistics: count, children-per-pair, effective pairs.

    ``pair_sizes`` is :attr:`MatingPairs.sizes`. Per-individual mate counts
    live in :func:`compute_aggregate_sections` under ``reproduction:`` (over
    **all** individuals, zero-included). This section is reserved for
    per-Mating-Pair quantities only.
    """
    n_pairs = len(pair_sizes)
    if n_pairs == 0:
        return None
    n_multi_child_pairs = int((pair_sizes >= 2).sum())

    return {
        "n_pairs": n_pairs,
        "n_pairs_with_multiple_children": n_multi_child_pairs,
        "frac_pairs_with_multiple_children": n_multi_child_pairs / n_pairs,
        "children_per_pair": _numeric_distribution(pair_sizes),
        "effective_pairs_by_children": _effective_count_from_weights(pair_sizes),
    }


def _offspring_dist(counts: np.ndarray, n: int) -> dict:
    if n == 0:
        return dict.fromkeys(("0", "1", "2", "3", "4+"), 0.0)
    out = {"0": float((counts == 0).sum()) / n}
    for k in (1, 2, 3):
        out[str(k)] = float((counts == k).sum()) / n
    out["4+"] = float((counts >= 4).sum()) / n
    return out


class _DepthRow(TypedDict):
    """One row of :func:`compute_founder_summary`'s ``by_depth`` table.

    Declared rather than left as a bare ``dict`` so per-key types survive
    lookup: an inferred ``dict[str, int | float | dict]`` collapses every
    ``row[...]`` to that union, which then fails comparison and ``min()``.
    """

    depth: int
    n: int
    active_founders: int
    active_founder_frac: float
    effective_founders_by_descendants: float
    founder_ancestors: dict


def compute_founder_summary(
    idf: pl.DataFrame,
    mother_rows: np.ndarray,
    father_rows: np.ndarray,
    max_lineage_cells: int = 5_000_000,
) -> tuple[dict, np.ndarray]:
    """Founder-contribution-by-depth using unique **Founder Ancestor** sets.

    ``mother_rows`` / ``father_rows`` give each row's parent row, ``-1`` when
    unknown.

    Returns a ``(summary, n_founder_ancestors)`` tuple. The second element
    is the per-individual count of distinct **Founder Ancestors** (zero
    for **Founders** themselves; zeros when the section is skipped).

    Bounded: carrying founder sets per row can become large on very large
    pedigrees, so the section reports ``computed: false`` instead of
    risking a memory blow-up.
    """
    n = len(idf)
    founders = idf["is_founder"].to_numpy().astype(bool)
    founder_rows = np.where(founders)[0]
    n_founders = len(founder_rows)
    zeros = np.zeros(n, dtype=np.int32)
    if n == 0 or n_founders == 0:
        return {"computed": True, "by_depth": [], "bottleneck": None}, zeros
    if n * n_founders > max_lineage_cells:
        skip = {
            "computed": False,
            "skip_reason": (
                f"n_individuals * n_founders = {n * n_founders} exceeds max_lineage_cells={max_lineage_cells}"
            ),
        }
        return skip, zeros

    row_to_founder = {int(row): i for i, row in enumerate(founder_rows)}
    has_mom = mother_rows >= 0
    has_dad = father_rows >= 0
    depth_arr = idf["ped_depth"].to_numpy()
    order = np.argsort(depth_arr, kind="stable")

    founder_sets: list[set[int]] = [set() for _ in range(n)]
    for i in order:
        i_int = int(i)
        if founders[i_int]:
            founder_sets[i_int] = {row_to_founder[i_int]}
            continue
        s: set[int] = set()
        if has_mom[i_int]:
            s.update(founder_sets[int(mother_rows[i_int])])
        if has_dad[i_int]:
            s.update(founder_sets[int(father_rows[i_int])])
        founder_sets[i_int] = s

    n_founder_ancestors = np.array(
        [len(fs) for fs in founder_sets],
        dtype=np.int32,
    )

    by_depth: list[_DepthRow] = []
    for depth in np.unique(depth_arr):
        rows = np.where(depth_arr == depth)[0]
        active: set[int] = set()
        counts = np.zeros(n_founders, dtype=np.int64)
        line_counts = np.zeros(len(rows), dtype=np.int32)
        for pos, row in enumerate(rows):
            fs = founder_sets[int(row)]
            line_counts[pos] = len(fs)
            if not fs:
                continue
            active.update(fs)
            counts[list(fs)] += 1
        active_counts = counts[counts > 0]
        by_depth.append(
            {
                "depth": int(depth),
                "n": len(rows),
                "active_founders": len(active),
                "active_founder_frac": len(active) / n_founders,
                "effective_founders_by_descendants": _effective_count_from_weights(active_counts),
                "founder_ancestors": _numeric_distribution(line_counts),
            }
        )

    nonempty = [row for row in by_depth if row["n"] > 0]
    if nonempty:
        min_active = min(row["active_founders"] for row in nonempty)
        min_eff = min(row["effective_founders_by_descendants"] for row in nonempty)
        bottleneck = {
            "min_active_founders": int(min_active),
            "min_active_founder_frac": min_active / n_founders,
            "min_active_depths": [int(row["depth"]) for row in nonempty if row["active_founders"] == min_active],
            "min_effective_founders_by_descendants": float(min_eff),
            "min_effective_depths": [
                int(row["depth"]) for row in nonempty if row["effective_founders_by_descendants"] == min_eff
            ],
        }
    else:
        bottleneck = None

    summary = {"computed": True, "by_depth": by_depth, "bottleneck": bottleneck}
    return summary, n_founder_ancestors


def compute_aggregate_sections(
    idf: pl.DataFrame,
    founder_summary: dict,
    include_inbreeding: bool,
) -> dict:
    """Pedigree-level aggregate sections derived from the individual table.

    ``founder_summary`` is the dict returned by :func:`compute_founder_summary`;
    it is computed beforehand so that its per-individual
    ``n_founder_ancestors`` vector can be added to ``idf`` before this
    function is called.
    """
    n = len(idf)
    if n == 0:
        return {
            "reproduction": {},
            "genealogy": {},
            "founder_contribution": {},
            "founder_summary": founder_summary,
            "components": {},
            "sex_summary": {},
            "depth_summary": [],
        }

    n_offspring_arr = idf["n_offspring"].to_numpy()
    reproductive = n_offspring_arr > 0
    n_reproductive = int(reproductive.sum())
    n_terminal = int((~reproductive).sum())
    founders = idf["is_founder"].to_numpy().astype(bool)
    descendant_paths_arr = idf["n_descendant_paths"].to_numpy()
    descendant_path_counts = descendant_paths_arr[founders]

    n_mates_arr = idf["n_mates"].to_numpy()
    sex_arr = idf["sex"].to_numpy()
    male_mask = sex_arr == SEX_MALE
    female_mask = sex_arr == SEX_FEMALE
    n_male = int(male_mask.sum())
    n_female = int(female_mask.sum())

    # frac_with_full_sib: fraction of individuals WITH BOTH PARENTS present
    # who share their (mother, father) with at least one other individual.
    mothers_arr = idf["mother"].to_numpy()
    fathers_arr = idf["father"].to_numpy()
    n_full_sibs_arr = idf["n_full_sibs"].to_numpy()
    both_parents = (mothers_arr != -1) & (fathers_arr != -1)
    n_both = int(both_parents.sum())
    if n_both:
        frac_with_full_sib = float((n_full_sibs_arr[both_parents] >= 1).sum()) / n_both
    else:
        frac_with_full_sib = 0.0

    # Per-Individual reproductive output: offspring counts, mate counts,
    # reproductive/terminal classification. Distributions follow the
    # CONTEXT.md naming convention: <noun>_count for summary stats;
    # <noun>_count_hist for binned PMF; _male / _female stratify either.
    reproduction = {
        "n_reproductive": n_reproductive,
        "frac_reproductive": n_reproductive / n,
        "n_terminal": n_terminal,
        "frac_terminal": n_terminal / n,
        "frac_with_full_sib": frac_with_full_sib,
        "offspring_count": _numeric_distribution(n_offspring_arr),
        "offspring_count_hist": _offspring_dist(n_offspring_arr, n),
        "offspring_count_hist_male": _offspring_dist(n_offspring_arr[male_mask], n_male),
        "offspring_count_hist_female": _offspring_dist(n_offspring_arr[female_mask], n_female),
        # mate_count_* are over ALL males / ALL females (zero-included), a
        # behavior change from 0.9's `female_mate_count` / `male_mate_count`
        # which only summed over parents-with-children. See CHANGELOG 0.10.0.
        "mate_count": _numeric_distribution(n_mates_arr),
        "mate_count_male": _numeric_distribution(n_mates_arr[male_mask]) if n_male else None,
        "mate_count_female": _numeric_distribution(n_mates_arr[female_mask]) if n_female else None,
    }

    # Per-individual ancestry. Asymmetric semantics today (paths for
    # descendants, distinct for ancestors) — see CONTEXT.md.
    genealogy = {
        "descendant_paths": _numeric_distribution(descendant_paths_arr),
    }
    if include_inbreeding:
        genealogy["distinct_ancestors"] = _numeric_distribution(idf["n_distinct_ancestors"])
    else:
        genealogy["distinct_ancestors"] = None

    n_founders = int(founders.sum())
    founders_with_desc = int((descendant_path_counts > 0).sum()) if n_founders else 0
    founder_contribution = {
        "n_founders_with_descendants": founders_with_desc,
        "n_founders_without_descendants": n_founders - founders_with_desc,
        "frac_founders_with_descendants": (founders_with_desc / n_founders) if n_founders else 0.0,
        "descendant_paths_per_founder": _numeric_distribution(descendant_path_counts),
        "effective_founders_by_descendant_paths": _effective_count_from_weights(descendant_path_counts),
    }
    _, sizes = np.unique(idf["component_id"].to_numpy(), return_counts=True)
    component_dist = _numeric_distribution(sizes)
    singletons = int((sizes == 1).sum())
    n_comp = len(sizes)
    components = {
        "singletons": singletons,
        "singletons_frac": singletons / n_comp if n_comp else 0.0,
        "size_dist": {
            "1": float((sizes == 1).sum()) / n_comp if n_comp else 0.0,
            "2": float((sizes == 2).sum()) / n_comp if n_comp else 0.0,
            "3-9": float(((sizes >= 3) & (sizes <= 9)).sum()) / n_comp if n_comp else 0.0,
            "10-99": float(((sizes >= 10) & (sizes <= 99)).sum()) / n_comp if n_comp else 0.0,
            "100+": float((sizes >= 100).sum()) / n_comp if n_comp else 0.0,
        },
        "component_size": component_dist,
    }

    F_arr = idf["F"].to_numpy() if "F" in idf.columns else None
    is_founder_arr = founders.astype(np.int64)
    depth_col = idf["ped_depth"].to_numpy()

    sex_summary = {}
    for label, code in (("female", SEX_FEMALE), ("male", SEX_MALE)):
        mask = sex_arr == code
        n_sub = int(mask.sum())
        if n_sub == 0:
            continue
        sub_off = n_offspring_arr[mask]
        sx_reproductive = sub_off > 0
        row = {
            "n": n_sub,
            "n_founders": int(is_founder_arr[mask].sum()),
            "n_reproductive": int(sx_reproductive.sum()),
            "frac_reproductive": float(sx_reproductive.sum()) / n_sub,
            "n_terminal": int((~sx_reproductive).sum()),
            "offspring_count": _numeric_distribution(sub_off),
            "mate_count": _numeric_distribution(n_mates_arr[mask]),
            "depth": _numeric_distribution(depth_col[mask]),
        }
        if include_inbreeding and F_arr is not None:
            row["F"] = _numeric_distribution(F_arr[mask])
            row["n_inbred"] = int((F_arr[mask] > INBRED_TOL).sum())
        sex_summary[label] = row

    n_distinct_anc_arr = idf["n_distinct_ancestors"].to_numpy() if "n_distinct_ancestors" in idf.columns else None

    depth_summary = []
    for depth in np.unique(depth_col):
        mask = depth_col == depth
        n_sub = int(mask.sum())
        sub_off = n_offspring_arr[mask]
        d_reproductive = sub_off > 0
        row = {
            "depth": int(depth),
            "n": n_sub,
            "n_male": int((sex_arr[mask] == SEX_MALE).sum()),
            "n_female": int((sex_arr[mask] == SEX_FEMALE).sum()),
            "n_founders": int(is_founder_arr[mask].sum()),
            "n_reproductive": int(d_reproductive.sum()),
            "frac_reproductive": float(d_reproductive.sum()) / n_sub,
            "n_terminal": int((~d_reproductive).sum()),
            "offspring_count": _numeric_distribution(sub_off),
            "offspring_count_hist": _offspring_dist(sub_off, n_sub),
            "mate_count": _numeric_distribution(n_mates_arr[mask]),
            "mean_distinct_ancestors": None,
            "mean_descendant_paths": float(descendant_paths_arr[mask].mean()),
        }
        if include_inbreeding and F_arr is not None and n_distinct_anc_arr is not None:
            row["mean_distinct_ancestors"] = float(n_distinct_anc_arr[mask].mean())
            row["mean_F"] = float(F_arr[mask].mean())
            row["max_F"] = float(F_arr[mask].max())
            row["n_inbred"] = int((F_arr[mask] > INBRED_TOL).sum())
        depth_summary.append(row)

    return {
        "reproduction": reproduction,
        "genealogy": genealogy,
        "founder_contribution": founder_contribution,
        "founder_summary": founder_summary,
        "components": components,
        "sex_summary": sex_summary,
        "depth_summary": depth_summary,
    }


def compute_sibship_sizes(sibship_sizes: np.ndarray) -> dict:
    """Per-**Sibship** size statistics.

    A **Sibship** is the offspring set of one **Mating Pair**, so
    ``sibship_sizes`` is :attr:`MatingPairs.sizes`. This function only emits
    per-Sibship aggregates; per-individual offspring counts (binned and
    sex-stratified) live in :func:`compute_aggregate_sections` under
    ``reproduction:``.
    """
    n_sib = len(sibship_sizes)
    if n_sib == 0:
        return {"empty": True}

    size_dist = {str(k): float((sibship_sizes == k).sum()) / n_sib for k in (1, 2, 3)}
    size_dist["4+"] = float((sibship_sizes >= 4).sum()) / n_sib

    return {
        "empty": False,
        "n_sibships": int(n_sib),
        "mean": float(sibship_sizes.mean()),
        "median": float(np.median(sibship_sizes)),
        "q1": float(np.quantile(sibship_sizes, 0.25)),
        "q3": float(np.quantile(sibship_sizes, 0.75)),
        "size_dist": size_dist,
    }


def compute_relationship_summary_from_burden(df: pl.DataFrame, burden: RelationshipBurden) -> dict:
    """Build the existing report from native O(N) degree counts."""
    n = len(df)
    n_possible = n * (n - 1) // 2
    n_related = sum(burden.category_counts.values())
    by_degree = {str(d): 0 for d in range(1, 6)}
    for code, count in burden.category_counts.items():
        degree = RELATIONSHIPS[code].degree
        if degree > 0:
            by_degree[str(degree)] += count

    per_person = burden.per_person
    closest = np.zeros(n, dtype=np.int8)
    distributions = {}
    for degree in range(5, 0, -1):
        values = per_person[:, degree - 1]
        closest[values > 0] = degree
        distributions[str(degree)] = _numeric_distribution(values)
    closest_dist = {"none": int((closest == 0).sum())}
    closest_dist.update({str(d): int((closest == d).sum()) for d in range(1, 6)})

    depth = df["ped_depth"].to_numpy()
    depth_rows = []
    for d in range(int(depth.max()) + 1 if n else 0):
        n_d = int((depth == d).sum())
        possible = n_d * (n_d - 1) // 2
        related = int(burden.same_depth_pairs[d])
        depth_rows.append(
            {
                "depth": d,
                "n": n_d,
                "n_individual_pairs": possible,
                "n_related_pairs": related,
                "n_unrelated_pairs": possible - related,
                "related_pair_density": related / possible if possible else 0.0,
            }
        )

    return {
        "computed": True,
        **({"max_degree": 5} if n else {}),
        "n_individual_pairs": n_possible,
        "n_related_pairs": n_related,
        "n_unrelated_pairs": n_possible - n_related,
        "related_pair_density": n_related / n_possible if n_possible else 0.0,
        "related_pairs_by_closest_degree": by_degree,
        "closest_relationship_per_individual": closest_dist,
        "relatives_by_degree": distributions,
        "relatives_total": _numeric_distribution(per_person.sum(axis=1, dtype=np.int64)),
        "related_pair_density_by_depth": depth_rows,
    }


def compute_effective_size(pg: PedigreeGraph, estimators: Iterable[str]) -> dict:
    """Run the named Ne estimators via ``pedigree_graph`` and return all eight as YAML-ready dicts.

    Thin wrapper around ``pedigree_graph.effective_size.estimate_effective_sizes``.
    An estimator outside ``estimators`` reports ``{ne: None, reason:
    "not_requested", ...}``; the same shape carries a genuine refusal (for
    example ``missing_metadata``), so every record has an ``ne`` and the
    reason is never lost.
    """
    from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS, estimate_effective_sizes

    raw = estimate_effective_sizes(pg, tuple(estimators))
    return {
        name: {"ne": None, **_normalise_effective_size_keys(raw[name].to_dict())}
        for name in ALL_EFFECTIVE_SIZE_ESTIMATORS
    }


def compute_individual_delta_f_for_reference(pg: PedigreeGraph, reference_rows: np.ndarray, column: str) -> dict:
    """``ne_individual_delta_f`` averaged over a caller-chosen reference subpopulation.

    Gutiérrez et al. 2008 average the individual increase in inbreeding over a
    **reference subpopulation**; ``compute_effective_size`` uses
    pedigree-graph's default, the last observed depth. ``reference_rows`` are
    graph rows (``pg`` was built from the same frame, so frame rows). The record
    has the same keys as the default one, plus ``reference_column``.
    """
    from pedigree_graph.effective_size import ne_individual_delta_f

    result = ne_individual_delta_f(pg, reference=reference_rows)
    return {"ne": None, **_normalise_effective_size_keys(result.to_dict()), "reference_column": column}


#: Upstream label / counter fields renamed on the way out. pedigree-graph
#: indexes its Ne records by *observed* label and calls that label a
#: generation; pedsum's CONTEXT.md reserves "generation" for a temporal
#: birth cohort and calls the topological label a **Depth**, which is what
#: ``PedigreeGraph.depth`` actually supplies here. ``transition_from`` /
#: ``transition_to`` keep their upstream names — they are pairs of the same
#: depth labels — and ``cohort_years`` keeps its name because it really is a
#: calendar year. Delete this map when pedigree-graph adopts depth terms.
_EFFECTIVE_SIZE_KEY_RENAMES: dict[str, str] = {
    "generations": "depths",
    "parent_generations": "parent_depths",
    "final_generation": "final_depth",
    "n_generations_used": "n_depths_used",
}


def _normalise_effective_size_keys(d: dict) -> dict:
    """Rename upstream generation-labelled keys to pedsum's depth terminology."""
    return {_EFFECTIVE_SIZE_KEY_RENAMES.get(k, k): v for k, v in d.items()}


def _build_inbreeding_summary(F: np.ndarray) -> dict:
    """Aggregate a per-individual F vector into the YAML-shaped summary.

    F itself is computed by ``pedigree_graph.PedigreeGraph.inbreeding()``
    (Meuwissen-Luo); pedsum no longer owns an F implementation.  This
    helper produces only the histogram / aggregate fields previously
    returned by pedsum's own since-deleted F implementation, and drops
    the ``memo_size`` diagnostic (which described the deleted algorithm
    and has no analogue in the upstream implementation).
    """
    n = len(F)
    inbred = F > INBRED_TOL
    n_inbred = int(inbred.sum())
    # First edge is INBRED_TOL (not 0.0) so the first range bucket is
    # (INBRED_TOL, 0.0625] — exactly complementary to the "0" bucket
    # (F <= INBRED_TOL). Starting at 0.0 double-counts F in (0, INBRED_TOL]
    # (in both the "0" bucket and the first range bucket), inflating the
    # histogram past 1.0. CONTEXT.md: values 0 <= F <= 1e-9 are the zero bucket.
    edges = [INBRED_TOL, 0.0625, 0.125, 0.25, 1.0]
    hist: dict[str, float] = {}
    hist["0"] = float((F <= INBRED_TOL).sum()) / n if n else 0.0
    for lo, hi in pairwise(edges):
        label = f"<{hi:g}"
        hist[label] = float(((lo < F) & (hi >= F)).sum()) / n if n else 0.0
    return {
        "n_inbred": n_inbred,
        "frac_inbred": n_inbred / n if n else 0.0,
        "mean_F": float(F.mean()) if n else 0.0,
        "max_F": float(F.max()) if n else 0.0,
        "hist": hist,
    }


def equivalent_complete_generations(df: pl.DataFrame, mother_rows: np.ndarray, father_rows: np.ndarray) -> np.ndarray:
    """Per-individual **Equivalent Complete Generations** (ECG), float64.

    Maignel, Boichard & Verrier 1996: the sum over every known ancestor of
    ``(1/2)^n``, where ``n`` is the number of generations separating it from the
    individual. An ancestor reached by several paths counts once per path. The
    sum satisfies the parent recurrence

        ECG_i = Σ over known parents p of (1 + ECG_p) / 2,

    so a founder has 0, a child of two founders has 1, and a child with one known
    founder parent has 0.5. The recurrence runs one depth at a time over
    ``ped_depth``: every parent sits at a lower depth than its child, so each
    depth's parents are final before that depth is computed.
    """
    m_row, f_row = mother_rows, father_rows
    has_mom, has_dad = m_row >= 0, f_row >= 0
    depth = df["ped_depth"].to_numpy()
    ecg = np.zeros(len(df), dtype=np.float64)
    order = np.argsort(depth, kind="stable")
    bounds = np.searchsorted(depth[order], np.arange(1, int(depth.max(initial=0)) + 2))
    for lo, hi in pairwise(bounds):
        rows = order[lo:hi]
        mom, dad = has_mom[rows], has_dad[rows]
        ecg[rows] = 0.5 * (
            np.where(mom, 1.0 + ecg[np.where(mom, m_row[rows], 0)], 0.0)
            + np.where(dad, 1.0 + ecg[np.where(dad, f_row[rows], 0)], 0.0)
        )
    return ecg


def build_individual_df(
    df: pl.DataFrame,
    mother_rows: np.ndarray,
    father_rows: np.ndarray,
    mating: MatingPairs,
    F: np.ndarray,
    n_distinct_ancestors: np.ndarray,
    n_descendant_paths: np.ndarray,
    component_labels: np.ndarray,
    sex_source: np.ndarray,
) -> pl.DataFrame:
    """Assemble per-individual table with the maximal column set.

    ``mother_rows`` / ``father_rows`` give each row's parent row, ``-1`` when
    unknown, and ``mating`` groups the same rows by **Mating Pair**.
    ``n_founder_ancestors`` is added by the caller after
    :func:`compute_founder_summary` runs against this table.

    ``n_descendant_paths`` is stored as ``Int64``: a path count over a deep
    or inbred pedigree grows multiplicatively and does not fit ``Int32``.
    ``n_distinct_ancestors`` is a distinct-individual count bounded by the
    row count, so it stays at the ``Int32`` pedigree-graph returns.
    """
    n = len(df)
    ids_arr = df["id"].to_numpy()
    mothers = df["mother"].to_numpy()
    fathers = df["father"].to_numpy()
    sex = df["sex"].to_numpy()
    gen = df["ped_depth"].to_numpy()

    has_mom = mother_rows >= 0
    has_dad = father_rows >= 0
    is_founder = ~(has_mom | has_dad)
    rows_m = np.where(has_mom)[0]
    rows_f = np.where(has_dad)[0]

    fs_count = np.zeros(n, dtype=np.int32)
    fs_count[mating.children] = mating.sizes[mating.pair] - 1

    children_of_mother = np.bincount(mother_rows[rows_m], minlength=n).astype(np.int32)
    children_of_father = np.bincount(father_rows[rows_f], minlength=n).astype(np.int32)
    n_off = children_of_mother + children_of_father

    # A half sib shares one parent: that parent's other children, less the full sibs.
    n_mhs = np.zeros(n, dtype=np.int32)
    n_mhs[rows_m] = children_of_mother[mother_rows[rows_m]] - 1 - fs_count[rows_m]
    n_phs = np.zeros(n, dtype=np.int32)
    n_phs[rows_f] = children_of_father[father_rows[rows_f]] - 1 - fs_count[rows_f]

    mates_as_mother = np.bincount(mating.mother_rows, minlength=n)
    n_mates = (mates_as_mother + np.bincount(mating.father_rows, minlength=n)).astype(np.int32)

    def _parent_of(rows: np.ndarray, parent_rows: np.ndarray) -> np.ndarray:
        return np.where(rows >= 0, parent_rows[rows], -1)

    grandparents = np.column_stack(
        [
            _parent_of(mother_rows, mother_rows),
            _parent_of(mother_rows, father_rows),
            _parent_of(father_rows, mother_rows),
            _parent_of(father_rows, father_rows),
        ]
    )
    n_gp = (grandparents >= 0).sum(axis=1).astype(np.int32)
    # A grandparent reached through both parents has the grandchild once.
    grandparents.sort(axis=1)
    distinct = np.ones(grandparents.shape, dtype=bool)
    distinct[:, 1:] = grandparents[:, 1:] != grandparents[:, :-1]
    n_gc = np.bincount(grandparents[distinct & (grandparents >= 0)], minlength=n).astype(np.int32)

    n_ua = np.zeros(n, dtype=np.int32)
    n_ua[rows_m] += fs_count[mother_rows[rows_m]]
    n_ua[rows_f] += fs_count[father_rows[rows_f]]

    noff_of_child = n_off[mating.children].astype(np.int64)
    pair_offspring = np.bincount(mating.pair, weights=noff_of_child).astype(np.int64)
    ua_offspring_sum = np.zeros(n, dtype=np.int32)
    ua_offspring_sum[mating.children] = (pair_offspring[mating.pair] - noff_of_child).astype(np.int32)

    n_fc = np.zeros(n, dtype=np.int32)
    n_fc[rows_m] += ua_offspring_sum[mother_rows[rows_m]]
    n_fc[rows_f] += ua_offspring_sum[father_rows[rows_f]]

    return pl.DataFrame(
        {
            "id": ids_arr,
            "sex": sex.astype(np.int8),
            "sex_source": sex_source.astype(str),
            "mother": mothers,
            "father": fathers,
            "ped_depth": gen.astype(np.int32),
            "is_founder": is_founder,
            "F": F.astype(np.float64),
            "n_full_sibs": fs_count,
            "n_mat_half_sibs": n_mhs,
            "n_pat_half_sibs": n_phs,
            "n_offspring": n_off,
            "n_mates": n_mates,
            "component_id": component_labels.astype(np.int32),
            "n_grandparents": n_gp,
            "n_grandchildren": n_gc,
            "n_uncles_aunts": n_ua,
            "n_first_cousins": n_fc,
            "n_distinct_ancestors": n_distinct_ancestors,
            "n_descendant_paths": n_descendant_paths.astype(np.int64),
        }
    )
