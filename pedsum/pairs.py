"""Relationship-pair enumeration and PedigreeGraph construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError

from pedsum.base import SEX_UNKNOWN, PedigreeError
from pedsum.pedigree_ops import IdIndex
from pedsum.redaction import null_below

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from pedsum.ids import IdLabels


def augment_pair_counts(named: Mapping[str, int | None]) -> dict:
    """Add ``PO`` (= MO + FO) and ``by_degree`` aggregates to a named-codes dict.

    Shared by the default counts and the ``--per-individual-burden`` category
    counts, so the YAML output schema is identical regardless of source.

    A ``None`` count means the code was not computed (pedigree-graph
    reports an unrequested code as ``None``, never as ``0``, e.g. past
    ``--max-degree``). A degree none of whose codes was computed is ``None``
    in ``by_degree``, so "not counted" never reads as "none found". ``PO``
    is ``None`` whenever either of ``MO`` / ``FO`` was not computed rather
    than silently reporting a half sum.
    """
    out: dict = {code: (None if count is None else int(count)) for code, count in named.items()}

    by_degree: dict[int, int | None] = dict.fromkeys(range(6))
    for code, count in out.items():
        if count is not None:
            degree = RELATIONSHIPS[code].degree
            by_degree[degree] = (by_degree[degree] or 0) + count

    mo = out.get("MO", 0)
    fo = out.get("FO", 0)
    out["PO"] = None if (mo is None or fo is None) else mo + fo
    out["by_degree"] = by_degree
    return out


def build_pedigree_graph(df: pl.DataFrame, labels: IdLabels | None = None) -> PedigreeGraph:
    """Compact arbitrary IDs to ``0..n-1`` and build a full ``PedigreeGraph``.

    Threads ``sex`` through to ``PedigreeGraph.from_arrays`` so downstream
    sex-aware estimators (Ne_sr, the sex-decomposed Ne_V quadrants, sex-
    stratified relationship-pair extraction) receive correct sex data
    rather than the silent zeros that the bare-arrays construction path
    would supply. When the df carries a ``birth_year`` column (populated
    by ``load_and_validate`` under ``--birth-year-col``), the array is
    threaded through as well so the Hill overlapping-generation estimator
    can build its cohort window.

    When the df carries a ``twin`` column (``--zygosity-col``: each row's
    MZ co-twin, -1 for none), it becomes ``twin_ids``, so co-twins count as
    ``MZ`` rather than ``FS``. The ``mz_links`` Check has made the links
    reciprocal pairs of full sibs of compatible sex, as pedigree-graph
    requires. Multiple births found from dates (``validate
    --twin-max-days``) carry no zygosity and are never linked.

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

    An id that is a mother of one row and a father of another is refused
    (pedigree-graph 0.15, ``parent_in_both_roles``); validate keeps one only
    under ``--allow-missing-sex`` or ``--no-sex-check``. That raises
    ``PedigreeError`` naming the id through ``labels``.
    """
    ids = df["id"].to_numpy()
    n = len(ids)
    new_ids = np.arange(n, dtype=np.int64)
    id_index = IdIndex(ids)

    def _remap(parents: np.ndarray) -> np.ndarray:
        return np.where(parents == -1, -1, id_index.get_indexer(parents)).astype(np.int64)

    birth_year = df["birth_year"].to_numpy().astype(np.int32) if "birth_year" in df.columns else None
    twin_ids = _remap(df["twin"].to_numpy()) if "twin" in df.columns else None
    try:
        return PedigreeGraph.from_arrays(
            ids=new_ids,
            mother_ids=_remap(df["mother"].to_numpy()),
            father_ids=_remap(df["father"].to_numpy()),
            sex=df["sex"].to_numpy().astype(np.int8),
            birth_year=birth_year,
            twin_ids=twin_ids,
        )
    except PedigreeValidationError as e:
        if e.code != "parent_in_both_roles":
            raise
        raise PedigreeError(_both_roles_message(df, e.fields, labels)) from e


def _both_roles_message(df: pl.DataFrame, fields: Mapping[str, object], labels: IdLabels | None) -> str:
    """Name the first id in both parent roles, by its input ID, and the repair for it.

    The graph's ids are ``df``'s rows, so the error's ids and rows index ``df``.
    """
    ids = df["id"].to_numpy()
    label = labels.label if labels is not None else str

    def name(key: str) -> str:
        row = fields[key]
        assert isinstance(row, int)
        return label(ids[row])

    parent = fields["parent_id"]
    assert isinstance(parent, int)
    if df["sex"][parent] == SEX_UNKNOWN:
        repair = "validate --drop-offending drops it"
    else:
        repair = (
            "validate --break-role-conflicts clears the links its recorded sex contradicts, "
            "and validate --drop-offending drops it"
        )
    return (
        f"id {name('parent_id')} is the mother of id {name('mother_of_id')} and the father of id "
        f"{name('father_of_id')} ({fields['parent_count']} id(s) in both parent roles), which pedigree-graph "
        f"refuses; {repair}"
    )


def redact_relationship_pairs(pairs: dict, min_cell: int, n_total: int) -> None:
    """``--safe-attempt``: null each small count, and each small ``by_degree`` count."""
    for code in list(pairs):
        if code == "by_degree":
            null_below(pairs["by_degree"], list(pairs["by_degree"]), min_cell)
        elif isinstance(pairs[code], int):
            null_below(pairs, (code,), min_cell)
