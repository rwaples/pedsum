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
    relatives          <- PedigreeGraph.relatives_per_person (pure pedigree structure)
    born_at_year       <- birth_year if available, else base_year + depth

The remaining columns need phenotype/affection/demography a pedigree does not
contain, so they are emitted as explicit null placeholders (nullable-integer
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
import polars as pl
from pedigree_graph import RELATIONSHIPS

from pedsum.progress import relationship_progress

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from pedigree_graph import PedigreeGraph, RelationshipPairs

#: Calendar-year offset used when a pedigree has no birth years:
#: ``born_at_year = base_year + depth`` (depth 0 = founders).
#: Matches ``fitace_epimight.constants.BASE_YEAR``.
BASE_YEAR = 1960


@dataclass(frozen=True)
class _EpiRel:
    """One EPIMIGHT relationship code mapped onto pedigree-graph pair codes.

    Attributes:
        code: the EPIMIGHT relationship code (the emitted ``relationship_kind``).
        pair_codes: constituent ``pedigree_graph`` pair codes summed for this
            relationship (e.g. ``HS`` = maternal + paternal half sibs).
        directional: when True the relationship is asymmetric: only the
            younger member is counted, and the pairs export keeps the block's
            junior-first orientation instead of canonicalizing. pedigree-graph
            supplies both (``relatives_per_person`` credits only the junior
            member of an asymmetric category), so nothing is reordered here.
    """

    code: str
    pair_codes: tuple[str, ...]
    directional: bool = False

    @property
    def kinship(self) -> float:
        """Nominal kinship coefficient, read from the first constituent pair code.

        Because ``FS`` folds in MZ twins (kinship 0.5), this is the
        dominant-case coefficient (0.25 for ``FS``), not exact for every
        constituent pair. Reading it from ``RELATIONSHIPS`` rather than
        restating it here means a change upstream cannot silently leave
        pedsum emitting a stale coefficient.
        """
        return RELATIONSHIPS[self.pair_codes[0]].nominal_kinship


_EPIMIGHT_RELS: tuple[_EpiRel, ...] = (
    _EpiRel("PO", ("MO", "FO"), directional=True),
    _EpiRel("FS", ("FS", "MZ")),
    _EpiRel("HS", ("MHS", "PHS")),
    _EpiRel("mHS", ("MHS",)),
    _EpiRel("pHS", ("PHS",)),
    _EpiRel("Av", ("Av",), directional=True),
    _EpiRel("1G", ("GP",), directional=True),
    _EpiRel("1C", ("1C",)),
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

#: Columns this skeleton cannot fill from a bare pedigree (emitted as nulls).
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
        ValueError: if no code is given, or any code is not a known EPIMIGHT
            relationship code.
    """
    requested = tuple(codes)
    if not requested:
        raise ValueError(f"no relationship codes given; valid codes: {', '.join(EPIMIGHT_RELATIONSHIP_ORDER)}")
    unknown = [c for c in requested if c not in _EPI_REGISTRY]
    if unknown:
        valid = ", ".join(EPIMIGHT_RELATIONSHIP_ORDER)
        raise ValueError(f"unknown relationship code(s) {unknown}; valid codes: {valid}")
    return requested


def _relationship_rows(all_pairs: RelationshipPairs, code: str) -> tuple[np.ndarray, np.ndarray]:
    """The ``(idx1, idx2)`` graph rows of every pair of one EPIMIGHT code.

    Concatenates the row arrays of the code's constituent pair codes, so one
    kinship call covers the whole kind. Orientation is the library's: an
    asymmetric block stores the junior role first (``MO``/``FO`` offspring,
    ``GP`` descendant, ``Av`` niece/nephew), which is exactly the ``idx1`` a
    directional EPIMIGHT kind charges. Re-deriving it from depth would disagree with the contract on a
    skipped-generation pedigree, where an aunt can sit at the same depth as
    her niece.
    """
    blocks = [all_pairs[pair_code] for pair_code in _EPI_REGISTRY[code].pair_codes]
    return np.concatenate([b.first_rows for b in blocks]), np.concatenate([b.second_rows for b in blocks])


def _born_at_year(df: pl.DataFrame, depths: np.ndarray, base_year: int) -> pl.Series:
    """Birth year per row: real ``birth_year`` if present (``-1`` → null), else derived.

    ``load_and_validate`` adds a ``birth_year`` column (int32, sentinel ``-1``)
    only under ``--birth-year-col``; otherwise the EPIMIGHT convention
    ``base_year + depth`` is used.
    """
    if "birth_year" in df.columns:
        raw = df["birth_year"].to_numpy()
        return (
            pl.DataFrame({"v": raw.astype(np.int32)})
            .select(pl.when(pl.col("v") == -1).then(None).otherwise(pl.col("v")).cast(pl.Int32).alias("born_at_year"))
            .to_series()
        )
    return pl.Series("born_at_year", (base_year + depths).astype(np.int32), dtype=pl.Int32)


def build_epimight_skeleton(
    df: pl.DataFrame,
    pg: PedigreeGraph,
    *,
    rels: tuple[str, ...] = EPIMIGHT_RELATIONSHIP_ORDER,
    disorder: str = "trait1",
    base_year: int = BASE_YEAR,
    drop_founders: bool = False,
) -> pl.DataFrame:
    """Build the EPIMIGHT long-form skeleton from a validated pedigree.

    Structural columns (``person_id``, ``relationship_kind``, ``relatives``,
    ``born_at_year``) are computed; phenotype columns are null placeholders.

    Args:
        df: the topologically-sorted frame returned by ``load_and_validate``
            (must carry an ``id`` column; ``birth_year`` is used when present).
        pg: a ``PedigreeGraph`` built from ``df`` (row order must match ``df``);
            its ``depth`` supplies ``born_at_year`` and the founder filter, and
            ``relatives_per_person`` the ``relatives`` counts without a pair list.
        rels: EPIMIGHT relationship codes to emit; rows come out sorted by code.
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
    depths = pg.depth
    person_id = df["id"].cast(pl.String).rename("person_id")
    born_at_year = _born_at_year(df, depths, base_year)

    # One engine pass over every constituent code; sum() folds a kind's codes.
    pair_codes = tuple(dict.fromkeys(pc for code in rels for pc in _EPI_REGISTRY[code].pair_codes))
    with relationship_progress("relatives_per_person") as progress:
        counts = pg.relatives_per_person(categories=pair_codes, progress=progress)

    # Reused placeholder columns — every block shares the same null columns.
    na_i8 = pl.Series("failure_status", [None] * n, dtype=pl.Int8)
    na_i16_time = pl.Series("failure_time", [None] * n, dtype=pl.Int16)
    na_i16_dead = pl.Series("dead_at_year", [None] * n, dtype=pl.Int16)
    na_i32 = pl.Series("relatives_diagnosed", [None] * n, dtype=pl.Int32)

    # Blocks go out in sorted kind order, which is the (relationship_kind,
    # disorder) sort: one kind per block and one disorder. Sorting the 8N-row
    # frame instead would set the process's peak memory.
    blocks: list[pl.DataFrame] = []
    for code in sorted(rels):
        relatives = counts.sum(_EPI_REGISTRY[code].pair_codes).astype(np.int32)
        blocks.append(
            pl.DataFrame(
                {
                    "person_id": person_id,
                    "disorder": pl.Series("disorder", [disorder] * n, dtype=pl.String),
                    "failure_status": na_i8,  # placeholder: needs affection
                    "failure_time": na_i16_time,  # placeholder: needs onset/censoring
                    "relationship_kind": pl.Series("relationship_kind", [code] * n, dtype=pl.String),
                    "relatives": relatives,
                    "relatives_diagnosed": na_i32,  # placeholder: needs relatives' affection
                    "born_at_year": born_at_year,
                    "dead_at_year": na_i16_dead,  # placeholder: needs death age
                }
            ).select(EPIMIGHT_COLUMNS)
        )

    out = pl.concat(blocks)

    if drop_founders:
        keep = person_id.to_numpy()[depths > depths.min()]
        out = out.filter(pl.col("person_id").is_in(keep.tolist()))

    return out


def iter_relative_pairs(
    df: pl.DataFrame,
    pg: PedigreeGraph,
    *,
    rels: tuple[str, ...] = EPIMIGHT_RELATIONSHIP_ORDER,
    exact_kinship: bool = False,
) -> Iterator[pl.DataFrame]:
    """Yield the relative pairs one relationship kind at a time, in output order.

    Kinds come out in sorted order, one frame per distinct code in ``rels``
    (empty when the kind has no pairs), each sorted by ``id1, id2``. Their
    concatenation is :func:`build_relative_pairs`, so a writer can append the
    frames as they come without holding every kind, or sorting the whole list,
    at once. See :func:`build_relative_pairs` for the columns and orientation.
    """
    rels = validate_relationship_codes(rels)
    ids = df["id"].to_numpy()
    kinds = sorted(dict.fromkeys(rels))
    # Only the categories these kinds read: the engine still resolves the closer
    # categories a selected one depends on, so each pair keeps its closest code.
    # The "memory" assembly lowers the --pairs peak by 18% at 20M rows for 5%
    # more wall time.
    with relationship_progress("relationship_pairs") as progress:
        all_pairs = pg.relationship_pairs(
            categories=dict.fromkeys(pc for code in kinds for pc in _EPI_REGISTRY[code].pair_codes),
            execution="memory",
            progress=progress,
        )
    columns = (*RELATIVE_PAIR_COLUMNS, "kinship_exact") if exact_kinship else RELATIVE_PAIR_COLUMNS

    for code in kinds:
        rel = _EPI_REGISTRY[code]
        idx1, idx2 = _relationship_rows(all_pairs, code)
        id1, id2 = ids[idx1], ids[idx2]
        if not rel.directional:
            # Symmetric: canonicalize so the smaller id comes first. pair_kinship
            # is symmetric, so this column swap leaves kinship_exact aligned.
            id1, id2 = np.minimum(id1, id2), np.maximum(id1, id2)
        data = {"id1": id1, "id2": id2}
        if exact_kinship:
            data["kinship_exact"] = np.asarray(pg.pair_kinship(idx1, idx2), dtype=np.float64)
        del idx1, idx2, id1, id2
        frame = (
            pl.DataFrame(data)
            .sort(["id1", "id2"], maintain_order=True)
            .with_columns(
                relationship_kind=pl.lit(code, dtype=pl.String),
                kinship=pl.lit(rel.kinship, dtype=pl.Float64),
            )
            .select(columns)
        )
        del data
        yield frame
        # Drop this kind before building the next, so at most one is held.
        del frame


def build_relative_pairs(
    df: pl.DataFrame,
    pg: PedigreeGraph,
    *,
    rels: tuple[str, ...] = EPIMIGHT_RELATIONSHIP_ORDER,
    exact_kinship: bool = False,
) -> pl.DataFrame:
    """Build the long list of relative pairs backing the skeleton's counts.

    One row per relative pair per relationship code, with columns
    ``id1, id2, relationship_kind, kinship``. For directional kinds (``PO``,
    ``Av``, ``1G``) ``id1`` is the younger member and ``id2`` the older relative;
    symmetric kinds are canonicalized so ``id1 < id2``. This concatenates
    :func:`iter_relative_pairs`; the CLI writes that iterator kind by kind
    instead, so it never holds the whole list.

    ``kinship`` is the **nominal** coefficient looked up by ``relationship_kind``
    (identical for every pair of a kind) — not computed from the pedigree. Because
    the EPIMIGHT codes overlap, a maternal half-sib pair is listed under both
    ``HS`` and ``mHS`` and MZ twins appear as ``FS``, so the nominal value is the
    dominant-case coefficient. With ``exact_kinship=True`` an extra
    ``kinship_exact`` column carries the **exact pedigree** kinship from
    ``pg.pair_kinship`` — inbreeding-, MZ-, and multi-path-aware, so it can
    exceed the nominal value (e.g. inbred sibs, double first cousins).
    pedigree-graph computes the kinship recurrence in **float32** (its pinned
    numerical contract); the column is widened to float64 for the export, so
    the values it carries are float32-origin and a float64 recomputation can
    differ in the low bits.

    Args:
        df: the frame from ``load_and_validate`` (provides the ``id`` column).
        pg: a ``PedigreeGraph`` built from ``df`` (row order must match ``df``);
            supplies the pair blocks and the exact kinship.
        rels: EPIMIGHT relationship codes to emit, in output order.
        exact_kinship: add a ``kinship_exact`` column with exact pedigree kinship.
            Runs the kinship recurrence over every pair (no ``n×n`` matrix), so
            cost scales with pair count × pedigree depth.

    Returns:
        A DataFrame of relative pairs sorted by ``relationship_kind``, ``id1``,
        ``id2``. Empty (with the right columns) when no pairs exist.
    """
    return pl.concat(iter_relative_pairs(df, pg, rels=rels, exact_kinship=exact_kinship))


def relationship_diagnostics(frame: pl.DataFrame, rels: tuple[str, ...]) -> list[tuple[str, int, float]]:
    """Per relationship code, ``(code, n_people_with_a_relative, mean_relatives)``."""
    diags: list[tuple[str, int, float]] = []
    for code in rels:
        relatives = frame.filter(pl.col("relationship_kind") == code)["relatives"].to_numpy()
        with_rel = relatives > 0
        n_with = int(with_rel.sum())
        mean_rel = float(relatives[with_rel].mean()) if n_with else 0.0
        diags.append((code, n_with, mean_rel))
    return diags
