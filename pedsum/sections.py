"""Per-section summary computations over the pedigree / individual table."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import polars as pl
import scipy.sparse.csgraph as csgraph
from pedigree_graph import REL_REGISTRY

if TYPE_CHECKING:
    import scipy.sparse as sp
    from pedigree_graph import PedigreeGraph

from pedsum.base import INBRED_TOL, SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from pedsum.pedigree_ops import IdIndex, _full_sib_groups, _grandparent_arrays, _parent_rows


def compute_size_structure(
    df: pl.DataFrame,
    children_csr: sp.csr_matrix | None,
) -> tuple[dict, np.ndarray]:
    """Counts, sex breakdown, generation depth, connected components.

    Returns (summary_dict, component_labels) where component_labels[i] is the
    connected-component id for row i.
    """
    n = len(df)
    has_mom = df["mother"].to_numpy() != -1
    has_dad = df["father"].to_numpy() != -1
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


def _mating_pair_sizes(df: pl.DataFrame) -> np.ndarray:
    """Children-per-(mother, father) group sizes over rows with both parents known."""
    mothers = df["mother"].to_numpy()
    fathers = df["father"].to_numpy()
    both_present = (mothers != -1) & (fathers != -1)
    if not both_present.any():
        return np.array([], dtype=np.int64)
    pair_keys = np.column_stack([mothers[both_present], fathers[both_present]])
    _, counts = np.unique(pair_keys, axis=0, return_counts=True)
    return counts


def compute_mating_pair_summary(df: pl.DataFrame) -> dict | None:
    """Aggregate per-Mating-Pair statistics: count, children-per-pair, effective pairs.

    Per-individual mate counts live in :func:`compute_aggregate_sections` under
    ``reproduction:`` (over **all** individuals, zero-included). This section is
    reserved for per-Mating-Pair quantities only.
    """
    pair_sizes = _mating_pair_sizes(df)
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
    max_lineage_cells: int = 5_000_000,
) -> tuple[dict, np.ndarray]:
    """Founder-contribution-by-depth using unique **Founder Ancestor** sets.

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
    id_index = IdIndex(idf["id"].to_numpy())
    mothers = idf["mother"].to_numpy()
    fathers = idf["father"].to_numpy()
    m_row, has_mom = _parent_rows(mothers, id_index)
    f_row, has_dad = _parent_rows(fathers, id_index)
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
            s.update(founder_sets[int(m_row[i_int])])
        if has_dad[i_int]:
            s.update(founder_sets[int(f_row[i_int])])
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


def compute_sibship_sizes(df: pl.DataFrame) -> dict:
    """Per-**Sibship** size statistics.

    A **Sibship** is the offspring set of one **Mating Pair**. This function
    only emits per-Sibship aggregates; per-individual offspring counts
    (binned and sex-stratified) live in :func:`compute_aggregate_sections`
    under ``reproduction:``.
    """
    sibship_sizes = _mating_pair_sizes(df)
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


def compute_relationship_summary(
    df: pl.DataFrame,
    pair_lists: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> dict:
    """Density and per-individual relationship-burden summaries.

    Pair-list-derived metrics are exact for the matrix engine. The BFS engine
    currently returns aggregate counts only, so these fields are unavailable
    there instead of being approximated from non-unique relationship counts.
    """
    n = len(df)
    n_possible = n * (n - 1) // 2
    if pair_lists is None:
        return {
            "computed": False,
            "skip_reason": "relationship pair lists are only available from the matrix engine",
            "n_individual_pairs": int(n_possible),
        }
    if n == 0:
        return {
            "computed": True,
            "n_individual_pairs": 0,
            "n_related_pairs": 0,
            "n_unrelated_pairs": 0,
            "related_pair_density": 0.0,
            "related_pairs_by_closest_degree": {str(d): 0 for d in range(1, 6)},
            "closest_relationship_per_individual": {"none": 0, **{str(d): 0 for d in range(1, 6)}},
            "relatives_by_degree": {str(d): _numeric_distribution(np.array([], dtype=np.int64)) for d in range(1, 6)},
            "relatives_total": _numeric_distribution(np.array([], dtype=np.int64)),
            "related_pair_density_by_depth": [],
        }

    keys_parts = []
    degree_parts = []
    for code, (a_raw, b_raw) in pair_lists.items():
        if code not in REL_REGISTRY:
            continue
        a = np.asarray(a_raw, dtype=np.int64)
        b = np.asarray(b_raw, dtype=np.int64)
        if len(a) == 0:
            continue
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keep = lo != hi
        if not keep.any():
            continue
        keys_parts.append(lo[keep] * n + hi[keep])
        degree_parts.append(
            np.full(int(keep.sum()), REL_REGISTRY[code].degree, dtype=np.int8),
        )

    if not keys_parts:
        closest_degree = np.zeros(n, dtype=np.int8)
        return {
            "computed": True,
            "n_individual_pairs": int(n_possible),
            "n_related_pairs": 0,
            "n_unrelated_pairs": int(n_possible),
            "related_pair_density": 0.0,
            "related_pairs_by_closest_degree": {str(d): 0 for d in range(1, 6)},
            "closest_relationship_per_individual": {
                "none": int((closest_degree == 0).sum()),
                **{str(d): 0 for d in range(1, 6)},
            },
            "relatives_by_degree": {str(d): _numeric_distribution(np.zeros(n, dtype=np.int64)) for d in range(1, 6)},
            "relatives_total": _numeric_distribution(np.zeros(n, dtype=np.int64)),
            "related_pair_density_by_depth": [],
        }

    keys = np.concatenate(keys_parts)
    degrees = np.concatenate(degree_parts)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    degrees = degrees[order]
    starts = np.concatenate(([0], np.where(np.diff(keys) != 0)[0] + 1))
    unique_keys = keys[starts]
    min_degrees = np.minimum.reduceat(degrees, starts)

    lo = unique_keys // n
    hi = unique_keys % n
    n_related = len(unique_keys)

    counts_by_degree = {}
    total_relatives = np.zeros(n, dtype=np.int64)
    closest_degree = np.zeros(n, dtype=np.int8)
    for degree in range(1, 6):
        mask = min_degrees == degree
        degree_counts = (np.bincount(lo[mask], minlength=n) + np.bincount(hi[mask], minlength=n)).astype(np.int64)
        counts_by_degree[str(degree)] = _numeric_distribution(degree_counts)
        total_relatives += degree_counts

    for degree in range(5, 0, -1):
        has_degree = (
            np.bincount(lo[min_degrees == degree], minlength=n) + np.bincount(hi[min_degrees == degree], minlength=n)
        ) > 0
        closest_degree[has_degree] = degree

    related_by_closest_degree = {str(degree): int((min_degrees == degree).sum()) for degree in range(1, 6)}
    closest_dist = {"none": int((closest_degree == 0).sum())}
    closest_dist.update({str(degree): int((closest_degree == degree).sum()) for degree in range(1, 6)})

    depth = df["ped_depth"].to_numpy()
    depth_rows = []
    for d in range(int(depth.max()) + 1 if n else 0):
        n_d = int((depth == d).sum())
        possible = n_d * (n_d - 1) // 2
        if possible:
            related = int(((depth[lo] == d) & (depth[hi] == d)).sum())
            density = related / possible
        else:
            related = 0
            density = 0.0
        depth_rows.append(
            {
                "depth": int(d),
                "n": n_d,
                "n_individual_pairs": int(possible),
                "n_related_pairs": related,
                "n_unrelated_pairs": int(possible - related),
                "related_pair_density": float(density),
            }
        )

    return {
        "computed": True,
        "max_degree": 5,
        "n_individual_pairs": int(n_possible),
        "n_related_pairs": n_related,
        "n_unrelated_pairs": int(n_possible - n_related),
        "related_pair_density": (n_related / n_possible) if n_possible else 0.0,
        "related_pairs_by_closest_degree": related_by_closest_degree,
        "closest_relationship_per_individual": closest_dist,
        "relatives_by_degree": counts_by_degree,
        "relatives_total": _numeric_distribution(total_relatives),
        "related_pair_density_by_depth": depth_rows,
    }


def compute_effective_size(
    pg: PedigreeGraph,
    *,
    ne_coancestry: bool = False,
    n_threads: int = 1,
) -> dict:
    """Run the eight pedigree-based Ne estimators via ``pedigree_graph``.

    Thin wrapper around ``pedigree_graph.compute_all_ne``: builds the
    founder-contribution structures once, dispatches every estimator,
    and serialises each result dataclass to a YAML-ready dict via its
    own ``.to_dict()`` method.

    When ``ne_coancestry`` is False (the default), the coancestry-rate
    Ne_C estimator is skipped — its kinship DP can dominate memory on
    very large pedigrees.  The ``ne_coancestry`` slot in the returned
    dict will then carry ``ne=None`` and NaN per-gen arrays.
    """
    from pedigree_graph import compute_all_ne

    raw = compute_all_ne(
        pg,
        skip_ne_coancestry=not ne_coancestry,
        n_threads=n_threads,
    )
    return {name: _normalise_effective_size_keys(result.to_dict()) for name, result in raw.items()}


def _normalise_effective_size_keys(d: dict) -> dict:
    """Rename upstream ``n_generations_used`` → ``n_depths_used`` on the way out.

    pedigree-graph emits ``n_generations_used`` in ``ne_inbreeding`` and
    ``ne_caballero_toro``; pedsum's CONTEXT.md treats "generation" as a
    temporal/birth-cohort term, distinct from topological **Depth**.
    This shim renames the key in pedsum's output without touching the
    upstream package. Delete this helper when pedigree-graph itself
    adopts the depth-based name.
    """
    if "n_generations_used" in d:
        d = {("n_depths_used" if k == "n_generations_used" else k): v for k, v in d.items()}
    return d


def _build_inbreeding_summary(F: np.ndarray) -> dict:
    """Aggregate a per-individual F vector into the YAML-shaped summary.

    F itself is computed by ``pedigree_graph.PedigreeGraph.compute_inbreeding()``
    (Meuwissen-Luo); pedsum no longer owns an F implementation.  This
    helper produces only the histogram / aggregate fields previously
    returned by the deleted ``compute_inbreeding`` function, and drops
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


def build_individual_df(
    df: pl.DataFrame,
    id_index: IdIndex,
    F: np.ndarray,
    n_distinct_ancestors: np.ndarray,
    n_descendant_paths: np.ndarray,
    component_labels: np.ndarray,
    sex_source: np.ndarray,
) -> pl.DataFrame:
    """Assemble per-individual table with the maximal column set.

    ``n_founder_ancestors`` is added by the caller after
    :func:`compute_founder_summary` runs against this table.
    """
    n = len(df)
    ids_arr = df["id"].to_numpy()
    mothers = df["mother"].to_numpy()
    fathers = df["father"].to_numpy()
    sex = df["sex"].to_numpy()
    gen = df["ped_depth"].to_numpy()

    m_row, has_mom = _parent_rows(mothers, id_index)
    f_row, has_dad = _parent_rows(fathers, id_index)
    is_founder = ~(has_mom | has_dad)

    fs_count64, _, both_present = _full_sib_groups(df)
    fs_count = fs_count64.astype(np.int32)
    rows_m = np.where(has_mom)[0]
    rows_f = np.where(has_dad)[0]
    rows_bp = np.where(both_present)[0]

    def _shared_parent_counts(parents: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Per-row count of OTHER children sharing this row's parent id."""
        _, inv, counts = np.unique(parents[rows], return_inverse=True, return_counts=True)
        return counts[inv].astype(np.int64) - 1

    n_mhs = np.zeros(n, dtype=np.int32)
    if rows_m.size:
        share_mom = _shared_parent_counts(mothers, rows_m)
        n_mhs[rows_m] = share_mom.astype(np.int32) - fs_count[rows_m]

    n_phs = np.zeros(n, dtype=np.int32)
    if rows_f.size:
        share_dad = _shared_parent_counts(fathers, rows_f)
        n_phs[rows_f] = share_dad.astype(np.int32) - fs_count[rows_f]

    n_off = np.bincount(m_row[has_mom], minlength=n).astype(np.int32) + np.bincount(f_row[has_dad], minlength=n).astype(
        np.int32
    )

    n_mates = np.zeros(n, dtype=np.int32)
    if rows_bp.size:
        bp_mothers = mothers[rows_bp]
        bp_fathers = fathers[rows_bp]
        pairs = np.unique(np.column_stack([bp_mothers, bp_fathers]), axis=0)
        mate_mothers, mates_per_mother = np.unique(pairs[:, 0], return_counts=True)
        mate_fathers, mates_per_father = np.unique(pairs[:, 1], return_counts=True)
        mom_rows = id_index.get_indexer(mate_mothers)
        n_mates[mom_rows] = mates_per_mother.astype(np.int32)
        dad_rows = id_index.get_indexer(mate_fathers)
        n_mates[dad_rows] += mates_per_father.astype(np.int32)

    mm, mf, fm, ff = _grandparent_arrays(df)
    n_gp = (
        (mm != -1).astype(np.int32)
        + (mf != -1).astype(np.int32)
        + (fm != -1).astype(np.int32)
        + (ff != -1).astype(np.int32)
    )

    js = np.tile(np.arange(n, dtype=np.int64), 4)
    gps = np.concatenate([mm, mf, fm, ff])
    keep = gps != -1
    if keep.any():
        gp_rows = id_index.get_indexer(gps[keep])
        unique_pairs = np.unique(np.column_stack([js[keep], gp_rows]), axis=0)
        n_gc = np.bincount(unique_pairs[:, 1], minlength=n).astype(np.int32)
    else:
        n_gc = np.zeros(n, dtype=np.int32)

    mother_fs = np.zeros(n, dtype=np.int32)
    if rows_m.size:
        mother_fs[rows_m] = fs_count[m_row[rows_m]]
    father_fs = np.zeros(n, dtype=np.int32)
    if rows_f.size:
        father_fs[rows_f] = fs_count[f_row[rows_f]]
    n_ua = mother_fs + father_fs

    ua_offspring_sum = np.zeros(n, dtype=np.int32)
    if rows_bp.size:
        noff_for_bp = n_off[rows_bp].astype(np.int64)
        pair_keys = np.column_stack([mothers[rows_bp], fathers[rows_bp]])
        _, inv = np.unique(pair_keys, axis=0, return_inverse=True)
        mating_total = np.bincount(inv, weights=noff_for_bp).astype(np.int64)
        per_child_total = mating_total[inv]
        ua_offspring_sum[rows_bp] = (per_child_total - noff_for_bp).astype(np.int32)

    n_fc = np.zeros(n, dtype=np.int32)
    if rows_m.size:
        n_fc[rows_m] += ua_offspring_sum[m_row[rows_m]]
    if rows_f.size:
        n_fc[rows_f] += ua_offspring_sum[f_row[rows_f]]

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
            "n_descendant_paths": n_descendant_paths.astype(np.int32),
        }
    )
