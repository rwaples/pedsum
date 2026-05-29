"""Relationship-pair enumeration and PedigreeGraph construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pedigree_graph import REL_REGISTRY, PedigreeGraph


def _augment_pair_counts(named: dict[str, int]) -> dict:
    """Add ``PO`` (= MO + FO) and ``by_degree`` aggregates to a named-codes dict.

    Shared by the matrix pair-list enumerator and the streaming-scalar
    counter so the YAML output schema is identical regardless of source.
    """
    out = {code: int(count) for code, count in named.items()}
    out["PO"] = int(named.get("MO", 0) + named.get("FO", 0))
    by_degree = dict.fromkeys(range(6), 0)
    for code, count in named.items():
        by_degree[REL_REGISTRY[code].degree] += int(count)
    out["by_degree"] = by_degree
    return out


def _count_pairs_matrix_with_lists(df: pd.DataFrame, pg: PedigreeGraph | None = None) -> dict:
    """Sparse matrix enumerator that retains pair lists for richer summaries.

    Delegates relationship enumeration to ``pedigree_graph.PedigreeGraph``;
    when ``pg`` is None this wrapper compacts IDs to ``0..n-1`` first
    because ``PedigreeGraph`` allocates an ``id_to_row`` table sized to
    ``max(id)+1``.

    Assumes every non-``-1`` parent ID appears in ``df['id']``; this is
    enforced by ``load_and_validate``'s ``parent_refs_present_*`` checks
    and lets the internal ID-compaction ``reindex`` skip NaN handling.

    When ``pg`` is supplied the wrapper reuses it instead of building a
    fresh compacted PedigreeGraph; saves one compaction pass when the
    caller already needed a graph for other primitives (F, lineage,
    effective size).
    """
    if pg is None:
        pg = _build_pedigree_graph(df)
    pair_lists = pg.extract_pairs(max_degree=5)
    named = {code: len(a) for code, (a, _) in pair_lists.items()}
    out = _augment_pair_counts(named)
    out["_pair_lists"] = pair_lists
    return out


def _build_pedigree_graph(df: pd.DataFrame) -> PedigreeGraph:
    """Compact arbitrary IDs to ``0..n-1`` and build a full ``PedigreeGraph``.

    Threads ``sex`` through to ``PedigreeGraph.from_arrays`` so downstream
    sex-aware estimators (Ne_sr, the sex-decomposed Ne_V quadrants, sex-
    stratified relationship-pair extraction) receive correct sex data
    rather than the silent zeros that the bare-arrays construction path
    would supply. When the df carries a ``birth_year`` column (populated
    by ``load_and_validate`` under ``--birth-year-col``), the array is
    threaded through as well so the Hill overlapping-generation estimator
    can build its cohort window.

    Generation is derived inside ``from_arrays`` from the (already-remapped)
    parent arrays via a fixed-point sweep — same semantics as pedsum's
    historical Kahn pass.  ``twin`` defaults to ``-1`` because pedsum's
    input format does not carry twin annotations.

    The compaction is necessary because ``PedigreeGraph`` allocates an
    ``id_to_row`` table sized to ``max(id) + 1``; passing original IDs
    on a sparse pedigree would inflate memory by orders of magnitude.

    Assumes the input has already passed ``load_and_validate``, which
    sorts rows into topological order; ``PedigreeGraph`` requires
    parents to precede children in row order.
    """
    ids = df["id"].to_numpy()
    n = len(ids)
    new_ids = np.arange(n, dtype=np.int64)
    id_to_compact = pd.Series(new_ids, index=ids)

    def _remap(parents: np.ndarray) -> np.ndarray:
        return np.where(
            parents == -1,
            -1,
            id_to_compact.reindex(parents).to_numpy(),
        ).astype(np.int64)

    birth_year = df["birth_year"].to_numpy().astype(np.int32) if "birth_year" in df.columns else None
    return PedigreeGraph.from_arrays(
        ids=new_ids,
        mothers=_remap(df["mother"].to_numpy()),
        fathers=_remap(df["father"].to_numpy()),
        sex=df["sex"].to_numpy().astype(np.int8),
        birth_year=birth_year,
    )
