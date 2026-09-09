"""Relationship-pair enumeration and PedigreeGraph construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pedigree_graph import RELATIONSHIPS, PedigreeGraph

from pedsum.pedigree_ops import IdIndex

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl


def _augment_pair_counts(named: Mapping[str, int | None]) -> dict:
    """Add ``PO`` (= MO + FO) and ``by_degree`` aggregates to a named-codes dict.

    Shared by the matrix pair-list enumerator and the streaming-scalar
    counter so the YAML output schema is identical regardless of source.

    A ``None`` count means the code was not computed (pedigree-graph
    reports an unrequested code as ``None``, never as ``0``). Such a code
    contributes nothing to ``by_degree``, and ``PO`` is ``None`` whenever
    either of ``MO`` / ``FO`` was not computed rather than silently
    reporting a half sum.
    """
    out: dict = {code: (None if count is None else int(count)) for code, count in named.items()}

    by_degree = dict.fromkeys(range(6), 0)
    for code, count in out.items():
        if count is not None:
            by_degree[RELATIONSHIPS[code].degree] += count

    mo = out.get("MO", 0)
    fo = out.get("FO", 0)
    out["PO"] = None if (mo is None or fo is None) else mo + fo
    out["by_degree"] = by_degree
    return out


def _count_pairs_matrix_with_lists(df: pl.DataFrame, pg: PedigreeGraph | None = None) -> dict:
    """Sparse matrix enumerator that retains pair lists for richer summaries.

    Delegates relationship enumeration to ``pedigree_graph.PedigreeGraph``.
    Every pair is assigned its single closest relationship category, so the
    23 counts partition the related pairs instead of overlapping.

    When ``pg`` is supplied the wrapper reuses it instead of building a
    fresh compacted PedigreeGraph; saves one compaction pass when the
    caller already needed a graph for other primitives (F, lineage,
    effective size).
    """
    if pg is None:
        pg = _build_pedigree_graph(df)
    pair_lists = pg.relationship_pairs(max_degree=5)
    named = {code: len(block) for code, block in pair_lists.items()}
    out = _augment_pair_counts(named)
    out["_pair_lists"] = pair_lists
    return out


def _build_pedigree_graph(df: pl.DataFrame) -> PedigreeGraph:
    """Compact arbitrary IDs to ``0..n-1`` and build a full ``PedigreeGraph``.

    Threads ``sex`` through to ``PedigreeGraph.from_arrays`` so downstream
    sex-aware estimators (Ne_sr, the sex-decomposed Ne_V quadrants, sex-
    stratified relationship-pair extraction) receive correct sex data
    rather than the silent zeros that the bare-arrays construction path
    would supply. When the df carries a ``birth_year`` column (populated
    by ``load_and_validate`` under ``--birth-year-col``), the array is
    threaded through as well so the Hill overlapping-generation estimator
    can build its cohort window.

    ``twin_ids`` is left unset because pedsum's input format does not
    carry twin annotations.

    pedigree-graph indexes its own ids densely and accepts rows in any
    order, so the compaction is no longer needed for memory or topology.
    It is retained because pedsum treats a compacted id and the graph row
    it names as the same number: ``pg`` returns *row* indices from
    ``relationship_pairs`` and the lineage kernels, and pedsum indexes
    ``df`` with them directly.

    Assumes the input has already passed ``load_and_validate``, which
    sorts rows into topological order and guarantees every non-``-1``
    parent ID appears in ``df['id']``; that lets the remap below skip
    missing-ID handling.
    """
    ids = df["id"].to_numpy()
    n = len(ids)
    new_ids = np.arange(n, dtype=np.int64)
    id_index = IdIndex(ids)

    def _remap(parents: np.ndarray) -> np.ndarray:
        return np.where(parents == -1, -1, id_index.get_indexer(parents)).astype(np.int64)

    birth_year = df["birth_year"].to_numpy().astype(np.int32) if "birth_year" in df.columns else None
    return PedigreeGraph.from_arrays(
        ids=new_ids,
        mother_ids=_remap(df["mother"].to_numpy()),
        father_ids=_remap(df["father"].to_numpy()),
        sex=df["sex"].to_numpy().astype(np.int8),
        birth_year=birth_year,
    )
