"""Build the structural skeleton of an EPIMIGHT long-form input from a pedigree.

EPIMIGHT (a liability-threshold ACE estimator) consumes a long-form
``pipeline_input.parquet``: one row per ``person × disorder × relationship_kind``
carrying, for each person, their count of relatives of that kind and how many of
those relatives are diagnosed, plus failure (affection) status/time and birth/
death years.

A pedigree carries only *structure*, so pedsum can fill exactly the columns that
are a pure function of the pedigree:

    person_id          <- id
    relationship_kind  <- the EPIMIGHT relationship codes (default EPIMIGHT_RELATIONSHIP_ORDER)
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
Keep the codes, constituent pair codes, directionality and default flags in sync.

``mAv`` and ``pAv`` split ``Av`` by the proband's mother or father. They are
not a union of pair codes. A proband's count is the parent's full-sib count,
read at the parent's row, so a parent's MZ co-twin (kinship ¼ with the
proband, not ⅛) is left out. A proband whose parents are related
(:data:`RELATED_PARENTS_KINSHIP`) gets neither half. This follows
``fitace_epimight.create_input._lineage_parent_rows`` (fitACE's ``multimight``
branch); keep the two in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from pedigree_graph import RELATIONSHIPS

from pedsum.base import logger
from pedsum.progress import relationship_progress

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from pedigree_graph import PedigreeGraph, RelationshipPairs

#: Calendar-year offset used when a pedigree has no birth years:
#: ``born_at_year = base_year + depth`` (depth 0 = founders).
#: Matches ``fitace_epimight.constants.BASE_YEAR``.
BASE_YEAR = 1960

#: The pair codes of a parent's sibs that a lineage split counts.
PARENT_SIB_PAIR_CODES: tuple[str, ...] = ("FS",)

#: Parents at least this kin (first cousins) give their children no ``mAv`` or
#: ``pAv``: their own relationship adds paths, and can make a parent's sib a
#: grandparent. Matches ``fitace_epimight.create_input.RELATED_PARENTS_KINSHIP``.
RELATED_PARENTS_KINSHIP = 1 / 16

Parent = Literal["mother", "father"]


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
        default: whether ``--rels`` includes the code when not given. ``MZ``,
            ``MO``, ``FO``, ``mAv`` and ``pAv`` are opt-in.
        via_parent: the parent of a lineage split. Its ``pair_codes`` then only
            name the unsplit kind; the count reads
            :data:`PARENT_SIB_PAIR_CODES` at that parent's row.
    """

    code: str
    pair_codes: tuple[str, ...]
    directional: bool = False
    default: bool = True
    via_parent: Parent | None = None

    @property
    def count_codes(self) -> tuple[str, ...]:
        """The pair codes the engine resolves for this kind."""
        return PARENT_SIB_PAIR_CODES if self.via_parent else self.pair_codes

    @property
    def kinship(self) -> float:
        """Nominal kinship coefficient, read from the first constituent pair code.

        Every constituent of a multi-code kind (``PO``, ``HS``) has the same
        nominal kinship. Reading it from ``RELATIONSHIPS`` rather than
        restating it here means a change upstream cannot silently leave
        pedsum emitting a stale coefficient.
        """
        return RELATIONSHIPS[self.pair_codes[0]].nominal_kinship


_EPIMIGHT_RELS: tuple[_EpiRel, ...] = (
    # MZ co-twins exist only under --zygosity-col; FS is the other full sibs.
    _EpiRel("MZ", ("MZ",), default=False),
    _EpiRel("PO", ("MO", "FO"), directional=True),
    # PO's maternal and paternal halves; EPIMIGHT 2.1 has no kind for either.
    _EpiRel("MO", ("MO",), directional=True, default=False),
    _EpiRel("FO", ("FO",), directional=True, default=False),
    _EpiRel("FS", ("FS",)),
    _EpiRel("HS", ("MHS", "PHS")),
    _EpiRel("mHS", ("MHS",)),
    _EpiRel("pHS", ("PHS",)),
    _EpiRel("Av", ("Av",), directional=True),
    _EpiRel("mAv", ("Av",), directional=True, default=False, via_parent="mother"),
    _EpiRel("pAv", ("Av",), directional=True, default=False, via_parent="father"),
    _EpiRel("1G", ("GP",), directional=True),
    _EpiRel("1C", ("1C",)),
)

#: Registry keyed by EPIMIGHT relationship code.
_EPI_REGISTRY: dict[str, _EpiRel] = {r.code: r for r in _EPIMIGHT_RELS}

#: Every EPIMIGHT relationship code, close to distant.
EPIMIGHT_RELATIONSHIP_CODES: tuple[str, ...] = tuple(r.code for r in _EPIMIGHT_RELS)

#: The default codes, in canonical close-to-distant order (matches the fitACE emitter).
EPIMIGHT_RELATIONSHIP_ORDER: tuple[str, ...] = tuple(r.code for r in _EPIMIGHT_RELS if r.default)

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

#: Column order of the relative-pairs export (iter_relative_pairs).
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
        raise ValueError(f"no relationship codes given; valid codes: {', '.join(EPIMIGHT_RELATIONSHIP_CODES)}")
    unknown = [c for c in requested if c not in _EPI_REGISTRY]
    if unknown:
        valid = ", ".join(EPIMIGHT_RELATIONSHIP_CODES)
        raise ValueError(f"unknown relationship code(s) {unknown}; valid codes: {valid}")
    return requested


def _relationship_rows(
    all_pairs: RelationshipPairs, code: str, parent_rows: dict[Parent, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The ``(idx1, idx2)`` graph rows of every pair of one EPIMIGHT code.

    Concatenates the row arrays of the code's constituent pair codes, so one
    kinship call covers the whole kind. Orientation is the library's: an
    asymmetric block stores the junior role first (``MO``/``FO`` offspring,
    ``GP`` descendant, ``Av`` niece/nephew), which is exactly the ``idx1`` a
    directional EPIMIGHT kind charges. Re-deriving it from depth would disagree with the contract on a
    skipped-generation pedigree, where an aunt can sit at the same depth as
    her niece.

    A lineage split pairs each child with its parent's sibs instead.
    """
    rel = _EPI_REGISTRY[code]
    blocks = [all_pairs[pair_code] for pair_code in rel.count_codes]
    first, second = np.concatenate([b.first_rows for b in blocks]), np.concatenate([b.second_rows for b in blocks])
    if rel.via_parent is None:
        return first, second
    return _children_of_sibs(first, second, parent_rows[rel.via_parent])


def _children_of_sibs(first: np.ndarray, second: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(child, parent's sib)`` rows for every child whose ``parent`` row is not -1.

    ``first``/``second`` list each sib pair once, in either order.
    """
    src, dst = np.concatenate([first, second]), np.concatenate([second, first])
    sibs = dst[np.argsort(src, kind="stable")]
    degree = np.bincount(src, minlength=len(parent))
    start = np.cumsum(degree) - degree
    child = np.flatnonzero(parent >= 0)
    of_child = parent[child]
    k = degree[of_child]
    # Entry j of child i's run reads sibs[start[parent of i] + j].
    offsets = np.repeat(start[of_child] - (np.cumsum(k) - k), k) + np.arange(int(k.sum()))
    return np.repeat(child, k), sibs[offsets]


def lineage_parent_rows(pg: PedigreeGraph, rels: Iterable[str]) -> dict[Parent, np.ndarray] | None:
    """Each graph row's mother and father row for the lineage split, -1 where it is skipped.

    None unless ``rels`` holds a lineage-split code (``mAv``, ``pAv``). Compute
    it once and pass it to both :func:`build_epimight_skeleton` and
    :func:`iter_relative_pairs`, which compute it themselves when given None.
    -1 where the parent is missing, and on both sides where the parents'
    kinship is at least :data:`RELATED_PARENTS_KINSHIP`. fitACE also matches
    recorded ids (sibs, half sibs, parent and child) to catch a parent outside
    its graph; ``load_and_validate`` puts every parent in pedsum's graph, so
    the kinship test already covers those cases (¼ or ⅛).
    """
    if not any(_EPI_REGISTRY[code].via_parent for code in validate_relationship_codes(rels)):
        return None
    mother, father = pg.mother_rows.astype(np.int32), pg.father_rows.astype(np.int32)
    related = np.zeros(len(mother), dtype=bool)
    both = (mother >= 0) & (father >= 0)
    # int64 key: mother * n overflows int32 beyond ~46k people.
    couples, inverse = np.unique(mother[both].astype(np.int64) * len(mother) + father[both], return_inverse=True)
    if len(couples):
        kin = np.asarray(pg.pair_kinship(*np.divmod(couples, len(mother))))
        related[both] = kin[inverse] >= RELATED_PARENTS_KINSHIP
    if n_related := int(related.sum()):
        logger.info("mAv/pAv: %d people of related parents get no aunts or uncles", n_related)
    return {"mother": np.where(related, -1, mother), "father": np.where(related, -1, father)}


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
    parent_rows: dict[Parent, np.ndarray] | None = None,
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
        parent_rows: :func:`lineage_parent_rows` for ``rels``, computed here
            when omitted.

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
    pair_codes = tuple(dict.fromkeys(pc for code in rels for pc in _EPI_REGISTRY[code].count_codes))
    with relationship_progress("relatives_per_person") as progress:
        counts = pg.relatives_per_person(categories=pair_codes, progress=progress)
    if parent_rows is None:
        parent_rows = lineage_parent_rows(pg, rels) or {}

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
        rel = _EPI_REGISTRY[code]
        relatives = counts.sum(rel.count_codes).astype(np.int32)
        if rel.via_parent:
            rows = parent_rows[rel.via_parent]
            relatives = np.where(rows >= 0, relatives[rows.clip(min=0)], 0).astype(np.int32)
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
    parent_rows: dict[Parent, np.ndarray] | None = None,
) -> Iterator[pl.DataFrame]:
    """Yield the relative pairs backing the skeleton's counts, one relationship kind at a time.

    Kinds come out in sorted order, one frame per distinct code in ``rels``
    (empty when the kind has no pairs), each sorted by ``id1, id2``, so a
    writer can append the frames as they come without holding every kind, or
    sorting the whole list, at once.

    One row per relative pair per relationship code, with columns
    ``id1, id2, relationship_kind, kinship``. For directional kinds (``PO``,
    ``MO``, ``FO``, ``Av``, ``mAv``, ``pAv``, ``1G``) ``id1`` is the younger member and ``id2`` the older relative;
    symmetric kinds are canonicalized so ``id1 < id2``.

    ``kinship`` is the **nominal** coefficient looked up by ``relationship_kind``
    (identical for every pair of a kind) — not computed from the pedigree. Because
    the EPIMIGHT codes overlap, a maternal half-sib pair is listed under both
    ``HS`` and ``mHS``. MZ co-twins (``--zygosity-col``) are ``MZ``, not
    ``FS``. With ``exact_kinship=True`` an extra
    ``kinship_exact`` column carries the **exact pedigree** kinship from
    ``pg.pair_kinship`` — inbreeding-, MZ-, and multi-path-aware, so it can
    exceed the nominal value (e.g. inbred sibs, double first cousins).
    pedigree-graph computes the kinship recurrence in **float32** (its pinned
    numerical contract); the column is widened to float64 for the export, so
    the values it carries are float32-origin and a float64 recomputation can
    differ in the low bits. Its cost scales with pair count × pedigree depth.

    ``df`` is the frame from ``load_and_validate`` and ``pg`` a
    ``PedigreeGraph`` built from it (row order must match ``df``).
    ``parent_rows`` is :func:`lineage_parent_rows` for ``rels``, computed here
    when omitted.
    """
    rels = validate_relationship_codes(rels)
    ids = df["id"].to_numpy()
    kinds = sorted(dict.fromkeys(rels))
    # Before the pair list exists, so the couples' kinship call does not add to its peak.
    if parent_rows is None:
        parent_rows = lineage_parent_rows(pg, kinds) or {}
    # Only the categories these kinds read: the engine still resolves the closer
    # categories a selected one depends on, so each pair keeps its closest code.
    # The "memory" assembly lowers the --pairs peak by 18% at 20M rows for 5%
    # more wall time.
    with relationship_progress("relationship_pairs") as progress:
        all_pairs = pg.relationship_pairs(
            categories=dict.fromkeys(pc for code in kinds for pc in _EPI_REGISTRY[code].count_codes),
            execution="memory",
            progress=progress,
        )
    columns = (*RELATIVE_PAIR_COLUMNS, "kinship_exact") if exact_kinship else RELATIVE_PAIR_COLUMNS

    for code in kinds:
        rel = _EPI_REGISTRY[code]
        idx1, idx2 = _relationship_rows(all_pairs, code, parent_rows)
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
