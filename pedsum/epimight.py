"""Build the structural skeleton of an EPIMIGHT long-form input from a pedigree.

EPIMIGHT (a liability-threshold ACE estimator) consumes a long-form
``pipeline_input.parquet``: one row per ``person × disorder × relationship_kind``
carrying, for each person, their count of relatives of that kind and how many of
those relatives are diagnosed, plus failure (affection) status/time and birth/
death years.

A pedigree carries only *structure*, so pedsum can fill exactly the columns that
are a pure function of the pedigree:

    person_id          <- id
    relationship_kind  <- the EPIMIGHT relationship codes (EPIMIGHT_RELATIONSHIP_ORDER)
    relatives          <- count_total_relatives (pure pedigree structure)
    born_at_year       <- birth_year if available, else base_year + generation

The remaining columns need phenotype/affection/demography a pedigree does not
contain, so they are emitted as explicit ``<NA>`` placeholders (nullable-integer
dtype — schema-correct and ready to fill):

    failure_status      (needs affection status)
    failure_time        (needs onset / censoring time)
    relatives_diagnosed (needs relatives' affection status)
    dead_at_year        (needs death age / demography)

The relationship grouping below mirrors fitACE's ``fitace.relationships``
(the EPIMIGHT-facing grouping over pedigree-graph pair codes) — pedsum is public
and cannot import the private ``fitace`` package, so this is a maintained copy.
Keep the codes, constituent pair codes, and directionality flags in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pedigree_graph import PedigreeGraph

#: Calendar-year offset used when a pedigree has no birth years:
#: ``born_at_year = base_year + generation`` (generation 0 = founders).
#: Matches ``fitace_epimight.constants.BASE_YEAR``.
BASE_YEAR = 1960


@dataclass(frozen=True)
class _EpiRel:
    """One EPIMIGHT relationship code mapped onto pedigree-graph pair codes.

    Attributes:
        code: the EPIMIGHT relationship code (the emitted ``relationship_kind``).
        pair_codes: constituent ``pedigree_graph`` pair codes summed for this
            relationship (e.g. ``HS`` = maternal + paternal half sibs).
        kinship: nominal kinship coefficient for the relationship. Because ``FS``
            folds in MZ twins (kinship 0.5), the value is the dominant-case
            coefficient (0.25 for ``FS``), not exact for every constituent pair.
        directional: when True the relationship is asymmetric and only the
            younger member is counted (``count_total_relatives`` unidirectional).
        orient_by_generation: when True the pair must be re-oriented so the
            younger member is ``idx1`` before counting (avuncular).
    """

    code: str
    pair_codes: tuple[str, ...]
    kinship: float
    directional: bool = False
    orient_by_generation: bool = False


_EPIMIGHT_RELS: tuple[_EpiRel, ...] = (
    _EpiRel("PO", ("MO", "FO"), 0.25, directional=True),
    _EpiRel("FS", ("FS", "MZ"), 0.25),
    _EpiRel("HS", ("MHS", "PHS"), 0.125),
    _EpiRel("mHS", ("MHS",), 0.125),
    _EpiRel("pHS", ("PHS",), 0.125),
    _EpiRel("Av", ("Av",), 0.125, directional=True, orient_by_generation=True),
    _EpiRel("1G", ("GP",), 0.125, directional=True),
    _EpiRel("1C", ("1C",), 0.0625),
)

#: Registry keyed by EPIMIGHT relationship code.
_EPI_REGISTRY: dict[str, _EpiRel] = {r.code: r for r in _EPIMIGHT_RELS}

#: Canonical close-to-distant relationship order (matches the fitACE emitter).
EPIMIGHT_RELATIONSHIP_ORDER: tuple[str, ...] = tuple(r.code for r in _EPIMIGHT_RELS)

#: Output column order (the EPIMIGHT Pipeline-input schema).
EPIMIGHT_COLUMNS: tuple[str, ...] = (
    "person_id",
    "disorder",
    "failure_status",
    "failure_time",
    "relationship_kind",
    "relatives",
    "relatives_diagnosed",
    "born_at_year",
    "dead_at_year",
)

#: Columns this skeleton cannot fill from a bare pedigree (emitted as <NA>).
PLACEHOLDER_COLUMNS: tuple[str, ...] = (
    "failure_status",
    "failure_time",
    "relatives_diagnosed",
    "dead_at_year",
)

#: Column order of the relative-pairs export (build_relative_pairs).
RELATIVE_PAIR_COLUMNS: tuple[str, ...] = ("id1", "id2", "relationship_kind", "kinship")


def validate_relationship_codes(codes: Iterable[str]) -> tuple[str, ...]:
    """Validate EPIMIGHT relationship codes, preserving the caller's order.

    Args:
        codes: candidate relationship codes.

    Returns:
        The codes as a tuple.

    Raises:
        ValueError: if any code is not a known EPIMIGHT relationship code.
    """
    requested = tuple(codes)
    unknown = [c for c in requested if c not in _EPI_REGISTRY]
    if unknown:
        valid = ", ".join(EPIMIGHT_RELATIONSHIP_ORDER)
        raise ValueError(f"unknown relationship code(s) {unknown}; valid codes: {valid}")
    return requested


def _orient_pairs_by_generation(
    blocks: tuple[tuple[np.ndarray, np.ndarray], ...],
    generations: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Re-orient each pair block so ``idx1`` is the younger member (higher generation)."""
    oriented: list[tuple[np.ndarray, np.ndarray]] = []
    for idx1, idx2 in blocks:
        if len(idx1) == 0:
            oriented.append((idx1, idx2))
            continue
        swap = generations[idx1] < generations[idx2]
        oriented.append((np.where(swap, idx2, idx1), np.where(swap, idx1, idx2)))
    return tuple(oriented)


def _relationship_pair_blocks(
    all_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    code: str,
    generations: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Materialize the ``(idx1, idx2)`` blocks for one EPIMIGHT relationship code.

    Gathers the blocks for each constituent pair code present in ``all_pairs``
    (a known code absent from ``all_pairs`` contributes nothing) and re-orients
    by generation when the relationship requires it (avuncular).
    """
    rel = _EPI_REGISTRY[code]
    blocks = tuple(all_pairs[pc] for pc in rel.pair_codes if pc in all_pairs)
    if rel.orient_by_generation:
        blocks = _orient_pairs_by_generation(blocks, generations)
    return blocks


def count_total_relatives(
    pair_blocks: tuple[tuple[np.ndarray, np.ndarray], ...],
    n: int,
    *,
    unidirectional: bool = False,
) -> np.ndarray:
    """Count, per person, their total relatives across the given pair blocks.

    Args:
        pair_blocks: ``(idx1, idx2)`` arrays for one relationship type.
        n: total number of individuals (output length).
        unidirectional: when True, only ``idx1`` accrues the relative (used for
            asymmetric relationships where ``idx1`` is the younger member).

    Returns:
        An ``int32`` array of per-person relative counts.
    """
    # int64 accumulation is faster than narrow ints for np.add.at on this kernel.
    counts = np.zeros(n, dtype=np.int64)
    for idx1, idx2 in pair_blocks:
        if len(idx1) == 0:
            continue
        np.add.at(counts, idx1, 1)
        if not unidirectional:
            np.add.at(counts, idx2, 1)
    return counts.astype(np.int32, copy=False)


def _born_at_year(df: pd.DataFrame, generations: np.ndarray, base_year: int) -> pd.api.extensions.ExtensionArray:
    """Birth year per row: real ``birth_year`` if present (``-1`` → ``<NA>``), else derived.

    ``load_and_validate`` adds a ``birth_year`` column (int32, sentinel ``-1``)
    only under ``--birth-year-col``; otherwise the EPIMIGHT convention
    ``base_year + generation`` is used.
    """
    if "birth_year" in df.columns:
        raw = df["birth_year"].to_numpy()
        out = pd.array(raw, dtype="Int32")
        out[raw == -1] = pd.NA
        return out
    return pd.array((base_year + generations).astype(np.int32), dtype="Int32")


def build_epimight_skeleton(
    df: pd.DataFrame,
    pg: PedigreeGraph,
    *,
    all_pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    rels: tuple[str, ...] = EPIMIGHT_RELATIONSHIP_ORDER,
    disorder: str = "trait1",
    base_year: int = BASE_YEAR,
    drop_founders: bool = False,
) -> pd.DataFrame:
    """Build the EPIMIGHT long-form skeleton from a validated pedigree.

    Structural columns (``person_id``, ``relationship_kind``, ``relatives``,
    ``born_at_year``) are computed; phenotype columns are ``<NA>`` placeholders.

    Args:
        df: the topologically-sorted frame returned by ``load_and_validate``
            (must carry an ``id`` column; ``birth_year`` is used when present).
        pg: a ``PedigreeGraph`` built from ``df`` (row order must match ``df``);
            its ``generation`` provides depth for orientation and ``born_at_year``.
        all_pairs: the dict from ``pg.extract_pairs()``; extracted here when None.
            Pass a precomputed dict to share one extraction with another consumer.
        rels: EPIMIGHT relationship codes to emit, in output order.
        disorder: the single ``disorder`` label emitted (one block per disorder
            in EPIMIGHT long form; a pedigree carries no trait).
        base_year: calendar offset for the derived ``born_at_year``.
        drop_founders: drop founder-generation rows. Off by default — this
            skeleton favors completeness; the fitACE emitter drops them because a
            founder's degenerate full-sib stratum breaks h² estimation, so opt in
            when the output feeds estimation.

    Returns:
        A DataFrame with the EPIMIGHT Pipeline-input columns, sorted by
        ``relationship_kind`` then ``disorder`` (contiguous rows per kind let the
        R driver skip Parquet row groups).
    """
    rels = validate_relationship_codes(rels)
    n = len(df)
    generations = np.asarray(pg.generation)
    person_id = df["id"].astype(str).to_numpy(dtype=object)
    born_at_year = _born_at_year(df, generations, base_year)

    if all_pairs is None:
        all_pairs = pg.extract_pairs()

    # Reused placeholder arrays — every block shares the same <NA> columns.
    na_i8 = pd.array([pd.NA] * n, dtype="Int8")
    na_i16 = pd.array([pd.NA] * n, dtype="Int16")
    na_i32 = pd.array([pd.NA] * n, dtype="Int32")

    blocks: list[pd.DataFrame] = []
    for code in rels:
        rel = _EPI_REGISTRY[code]
        pair_blocks = _relationship_pair_blocks(all_pairs, code, generations)
        relatives = count_total_relatives(pair_blocks, n, unidirectional=rel.directional)
        blocks.append(
            pd.DataFrame(
                {
                    "person_id": person_id,
                    "disorder": disorder,
                    "failure_status": na_i8,  # placeholder: needs affection
                    "failure_time": na_i16,  # placeholder: needs onset/censoring
                    "relationship_kind": code,
                    "relatives": relatives,
                    "relatives_diagnosed": na_i32,  # placeholder: needs relatives' affection
                    "born_at_year": born_at_year,
                    "dead_at_year": na_i16,  # placeholder: needs death age
                },
                columns=EPIMIGHT_COLUMNS,
            )
        )

    out = pd.concat(blocks, ignore_index=True)

    if drop_founders:
        keep = person_id[generations > generations.min()]
        out = out[out["person_id"].isin(keep)]

    return out.sort_values(["relationship_kind", "disorder"], kind="stable", ignore_index=True)


def build_relative_pairs(
    df: pd.DataFrame,
    pg: PedigreeGraph,
    *,
    all_pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    rels: tuple[str, ...] = EPIMIGHT_RELATIONSHIP_ORDER,
    exact_kinship: bool = False,
) -> pd.DataFrame:
    """Build the long list of relative pairs backing the skeleton's counts.

    One row per relative pair per relationship code, with columns
    ``id1, id2, relationship_kind, kinship``. For directional kinds (``PO``,
    ``Av``, ``1G``) ``id1`` is the younger member and ``id2`` the older relative;
    symmetric kinds are canonicalized so ``id1 < id2``.

    ``kinship`` is the **nominal** coefficient looked up by ``relationship_kind``
    (identical for every pair of a kind) — not computed from the pedigree. Because
    the EPIMIGHT codes overlap, a maternal half-sib pair is listed under both
    ``HS`` and ``mHS`` and MZ twins appear as ``FS``, so the nominal value is the
    dominant-case coefficient. With ``exact_kinship=True`` an extra
    ``kinship_exact`` column carries the **exact pedigree** kinship from
    ``pg.compute_pair_kinship`` — inbreeding-, MZ-, and multi-path-aware, so it can
    exceed the nominal value (e.g. inbred sibs, double first cousins).

    Args:
        df: the frame from ``load_and_validate`` (provides the ``id`` column).
        pg: a ``PedigreeGraph`` built from ``df`` (its ``generation`` orients pairs).
        all_pairs: the dict from ``pg.extract_pairs()``; extracted here when None.
        rels: EPIMIGHT relationship codes to emit, in output order.
        exact_kinship: add a ``kinship_exact`` column with exact pedigree kinship.
            Runs the kinship recurrence over every pair (no ``n×n`` matrix), so
            cost scales with pair count × pedigree depth.

    Returns:
        A DataFrame of relative pairs sorted by ``relationship_kind``, ``id1``,
        ``id2``. Empty (with the right columns) when no pairs exist.
    """
    rels = validate_relationship_codes(rels)
    ids = df["id"].to_numpy()
    generations = np.asarray(pg.generation)
    if all_pairs is None:
        all_pairs = pg.extract_pairs()
    columns = (*RELATIVE_PAIR_COLUMNS, "kinship_exact") if exact_kinship else RELATIVE_PAIR_COLUMNS

    # First pass: the (idx1, idx2) row indices per code. Computing exact kinship
    # over all codes in one call lets compute_pair_kinship share its DP work.
    code_idx: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code in rels:
        pair_blocks = [b for b in _relationship_pair_blocks(all_pairs, code, generations) if len(b[0])]
        if pair_blocks:
            idx1 = np.concatenate([a for a, _ in pair_blocks])
            idx2 = np.concatenate([b for _, b in pair_blocks])
            code_idx[code] = (idx1, idx2)
    exact = pg.compute_pair_kinship(code_idx) if (exact_kinship and code_idx) else {}

    blocks: list[pd.DataFrame] = []
    for code, (idx1, idx2) in code_idx.items():
        id1, id2 = ids[idx1], ids[idx2]
        if not _EPI_REGISTRY[code].directional:
            # Symmetric: canonicalize so the smaller id comes first. compute_pair_kinship
            # is symmetric, so this column swap leaves kinship_exact aligned.
            id1, id2 = np.minimum(id1, id2), np.maximum(id1, id2)
        data = {"id1": id1, "id2": id2, "relationship_kind": code, "kinship": _EPI_REGISTRY[code].kinship}
        if exact_kinship:
            data["kinship_exact"] = exact[code]
        blocks.append(pd.DataFrame(data, columns=columns))

    if not blocks:
        return pd.DataFrame(columns=columns)
    out = pd.concat(blocks, ignore_index=True)
    return out.sort_values(["relationship_kind", "id1", "id2"], kind="stable", ignore_index=True)


def relationship_diagnostics(frame: pd.DataFrame, rels: tuple[str, ...]) -> list[tuple[str, int, float]]:
    """Per relationship code, ``(code, n_people_with_a_relative, mean_relatives)``."""
    diags: list[tuple[str, int, float]] = []
    for code in rels:
        block = frame[frame["relationship_kind"] == code]
        with_rel = block["relatives"] > 0
        n_with = int(with_rel.sum())
        mean_rel = float(block.loc[with_rel, "relatives"].mean()) if n_with else 0.0
        diags.append((code, n_with, mean_rel))
    return diags
