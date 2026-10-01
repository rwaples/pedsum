"""Pair-list oracle for pedsum's relationship-burden report.

Production builds the report from native burden arrays; the parity tests in
``test_relationship_summary_properties.py`` check it against this fold over
explicit pair lists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pedigree_graph import RELATIONSHIPS

from pedsum.sections import _numeric_distribution

if TYPE_CHECKING:
    import polars as pl


def compute_relationship_summary(
    df: pl.DataFrame,
    pair_lists: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> dict:
    """Fold pair lists into the relationship-burden report, one pair at a time.

    pedsum's ``--per-individual-burden`` path builds this report from
    pedigree-graph's O(N) burden arrays
    (``compute_relationship_summary_from_burden``). This independent
    pair-list fold is the parity oracle for it: each pair is deduplicated
    and charged its closest degree here, in Python, rather than by the engine.
    """
    n = len(df)
    n_possible = n * (n - 1) // 2
    if pair_lists is None:
        return {
            "computed": False,
            "skip_reason": "relationship burden requires --per-individual-burden",
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
    for code, block in pair_lists.items():
        if code not in RELATIONSHIPS:
            continue
        first_rows, second_rows = block
        a = np.asarray(first_rows, dtype=np.int64)
        b = np.asarray(second_rows, dtype=np.int64)
        if len(a) == 0:
            continue
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keep = lo != hi
        if not keep.any():
            continue
        keys_parts.append(lo[keep] * n + hi[keep])
        degree_parts.append(
            np.full(int(keep.sum()), RELATIONSHIPS[code].degree, dtype=np.int8),
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
