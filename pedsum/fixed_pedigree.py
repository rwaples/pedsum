"""The Fixed Pedigree: the validated pedigree with pedsum's auto-fixes applied.

``validate`` writes it as ``validate.tsv.gz``; ``summarize`` and the other
commands read the same pedigree through ``load_and_validate``
(``docs/adr/0005-check-fail-policy.md``). :func:`fix_pedigree` builds it from a
:class:`~pedsum.validate.ValidationContext` that has no Blocking Check
failures, and :func:`render_fixed_pedigree` turns it into the string frame
``validate`` writes. Writing the file stays with the caller.

The auto-fixes:

- a founder row for each missing parent ID (``parent_refs_present_*``);
- with ``fill_half_founders``, a phantom parent in each half-founder's missing
  slot;
- sex resolved from parent role, recorded per row in ``sex_source`` (an
  input that has the column, such as ``validate.tsv.gz``, keeps its values
  where this run changed nothing);
- rows in topological order (parents before children).

With ``--twin-max-days`` it also writes each row's multiple-birth set in a
``multiple_birth_set`` column (an annotation, not a fix). With
``--multiple-birth-col`` that column holds the sets pedsum read from the named
column, renumbered: missing tokens are no set and ids equal after stripping
whitespace are one set, so a reader sees the sets the Checks saw (pedsum#29).
Without either flag an input's ``multiple_birth_set`` column is dropped: rows
may have changed, and the sets cannot be recomputed. With
``--multiple-birth-col`` / ``--zygosity-col`` it clears the set and zygosity
cells the ``multiple_birth_sets`` and ``mz_links`` Checks flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, SexSource, logger
from pedsum.checks import rows_phrase
from pedsum.parse import plink_sex
from pedsum.pedigree_ops import unique_ints

if TYPE_CHECKING:
    from collections.abc import Callable

    from pedsum.ids import IdLabels
    from pedsum.pedigree_ops import IdIndex
    from pedsum.validate import ValidationContext


@dataclass(frozen=True)
class FixedPedigree:
    """The input rows after the auto-fixes, plus the founder rows pedsum synthesized.

    Every per-row array is in output order (parents before children) and covers
    the input rows only; ``founders`` are written ahead of them.
    """

    ids: np.ndarray  # int64 ID codes
    sex: np.ndarray  # int8 resolved sex (SEX_MALE / SEX_FEMALE / SEX_UNKNOWN)
    mothers: np.ndarray  # int64 codes, -1 missing; phantom parents filled in when requested
    fathers: np.ndarray
    sex_source: np.ndarray  # where each row's sex came from (object array of str)
    birth_year: np.ndarray | None  # int32, sentinel -1; None without a birth-year column
    multiple_birth_set: np.ndarray | None  # int32 set number, -1 for none; None without --twin-max-days
    # --multiple-birth-col: each row's set code (-1 none) and whether its cell was cleared.
    multiple_birth: np.ndarray | None
    cleared_multiple_birth: np.ndarray | None
    # --zygosity-col: each row's MZ co-twin ID code (-1 none) and whether its cell was cleared.
    co_twin: np.ndarray | None
    cleared_zygosity: np.ndarray | None
    ped_depth: np.ndarray  # int32 topological depth, founders 0
    input_row: np.ndarray  # each row's position in the input
    reordered: bool  # False when the input was already parents-first (input_row is the identity)
    filled_mother: np.ndarray  # bool: the mother slot holds a phantom parent
    filled_father: np.ndarray
    # Synthesized founder rows ({"id", "sex"[, "reason"]}): missing parents by ID, then phantoms.
    founders: list[dict]
    labels: IdLabels

    @cached_property
    def written_sets(self) -> np.ndarray | None:
        """The ``multiple_birth_set`` column ``validate`` writes: set numbers 1, 2, ..., -1 for none.

        ``--twin-max-days``'s sets, or ``--multiple-birth-col``'s set codes
        renumbered in order of each set's first row (pedsum#29); None with neither.
        """
        if self.multiple_birth_set is not None or self.multiple_birth is None:
            return self.multiple_birth_set
        codes = self.multiple_birth
        sets = np.full(len(codes), -1, dtype=np.int32)
        rows = np.flatnonzero(codes >= 0)
        _, first, inverse = np.unique(codes[rows], return_index=True, return_inverse=True)
        rank = np.empty(len(first), dtype=np.int32)
        rank[np.argsort(first, kind="stable")] = np.arange(1, len(first) + 1, dtype=np.int32)
        sets[rows] = rank[inverse]
        return sets

    def frame(self) -> pl.DataFrame:
        """The input rows as ``load_and_validate`` returns them: ID codes, int8 sex, ``ped_depth``.

        With ``--multiple-birth-col`` also ``multiple_birth`` (set code, -1
        none), and with ``--zygosity-col`` ``twin`` (MZ co-twin ID code, -1 none).
        """
        columns = {
            "id": self.ids,
            "sex": self.sex,
            "mother": self.mothers,
            "father": self.fathers,
            "sex_source": self.sex_source,
        }
        if self.birth_year is not None:
            columns["birth_year"] = self.birth_year
        if self.multiple_birth is not None:
            columns["multiple_birth"] = self.multiple_birth
        if self.co_twin is not None:
            columns["twin"] = self.co_twin
        columns["ped_depth"] = self.ped_depth
        # From the object array directly: a numpy unicode (astype(str)) column converts ~8x slower.
        return pl.DataFrame(columns, schema_overrides={"sex_source": pl.String})


def fix_pedigree(
    ctx: ValidationContext,
    *,
    fill_half_founders: bool = False,
    next_id: int | None = None,
    synthesize_founders: bool = True,
) -> FixedPedigree:
    """Apply the auto-fixes to a validated context.

    ``ctx`` must have parsed IDs and have no Blocking Check failures.
    ``next_id`` numbers the phantom parents; it defaults to one past the
    largest ID in ``ctx``, and ``--drop-offending`` passes the input's so no
    phantom reuses a dropped ID. ``synthesize_founders=False`` skips the scan
    for missing parents, for a caller whose Checks all passed and so has none.
    """
    ids, mothers, fathers = ctx.require_id_parents()
    assert ctx.id_labels is not None
    labels = ctx.id_labels
    imp = ctx.imputation
    _log_sex_imputation(imp, allow_missing_sex=ctx.config.allow_missing_sex)

    n = len(ids)
    max_days = ctx.config.twin_max_days
    if max_days is not None:
        assert ctx.birth_date is not None, "--twin-max-days needs the parsed birth dates"
        sets = multiple_birth_sets(mothers, fathers, ctx.birth_date, max_days)
    founders = (
        _build_added_founders(mothers, fathers, ctx.id_index, ctx.config.no_sex_check, ctx.rows)
        if synthesize_founders
        else []
    )
    filled_mother = filled_father = np.zeros(n, dtype=bool)
    if fill_half_founders:
        if next_id is None:
            next_id = next_free_id(ids, mothers, fathers, ctx.config.missing_parent_tokens)
        labels = labels.with_phantoms(next_id)
        new_mothers, new_fathers, phantoms = _build_phantom_parents(mothers, fathers, next_id)
        filled_mother = new_mothers != mothers
        filled_father = new_fathers != fathers
        mothers, fathers = new_mothers, new_fathers
        founders = founders + phantoms
        logger.info("added %d phantom parent(s) for half-founders", len(phantoms))

    # Parents before children, by the depth the acyclic Check computed. A
    # cycle BLOCKs or is dropped before a pedigree is fixed (ADR 0005).
    depth = ctx.depth
    assert (depth >= 0).all(), "fix_pedigree needs an acyclic pedigree"
    order = np.argsort(depth, kind="stable")
    n_moved = int((order != np.arange(n)).sum())
    if n_moved:
        logger.info("reordering %d/%d row(s) into topological order (parents before children)", n_moved, n)

    def take(values: np.ndarray) -> np.ndarray:
        # Already parents-first (the usual case for pedsum's own output): no copy.
        return values[order] if n_moved else values

    return FixedPedigree(
        ids=take(ids),
        sex=take(imp.imputed_sex),
        mothers=take(mothers),
        fathers=take(fathers),
        sex_source=take(_keep_recorded_sex_source(imp.sex_source, ctx.df_raw)),
        birth_year=None if ctx.birth_year is None else take(ctx.birth_year),
        multiple_birth_set=None if max_days is None else take(sets),
        multiple_birth=None if ctx.multiple_birth is None else take(ctx.multiple_birth),
        cleared_multiple_birth=None if ctx.cleared_multiple_birth is None else take(ctx.cleared_multiple_birth),
        co_twin=None if ctx.co_twin is None else take(ctx.co_twin),
        cleared_zygosity=None if ctx.cleared_zygosity is None else take(ctx.cleared_zygosity),
        ped_depth=take(depth),
        input_row=order,
        reordered=n_moved > 0,
        filled_mother=take(filled_mother),
        filled_father=take(filled_father),
        founders=founders,
        labels=labels,
    )


def render_fixed_pedigree(
    fp: FixedPedigree,
    raw: pl.DataFrame,
    *,
    id_col: str,
    sex_col: str,
    mother_col: str,
    father_col: str,
    missing_parent_tokens: tuple[str, ...] = (),
    multiple_birth_col: str | None = None,
    zygosity_col: str | None = None,
) -> pl.DataFrame:
    """The string frame ``validate`` writes: the input's columns and tokens, with the fixes.

    ``raw`` is the input as read (all String columns). Each input row keeps
    its own tokens except the sex column, written in PLINK's coding
    (1=male, 2=female, 0=unknown) whatever tokens the input used, a parent
    written as one of ``missing_parent_tokens``, written blank so the file
    reads the same without ``--missing-parent-token``, and a parent slot that
    now holds a phantom. A ``sex_source`` column records which rows pedsum
    changed, and a ``multiple_birth_set`` column, when ``fp`` has the sets,
    each row's set number (blank for none); otherwise an input column of that
    name is dropped. An input ``multiple_birth_set`` column holding other
    values is replaced with a WARNING, unless ``--twin-max-days`` recomputes
    it. The set and zygosity cells the Checks cleared are written blank.
    Founder rows come first, with empty cells outside the ID, sex and parent
    columns.
    """
    out = (raw[fp.input_row] if fp.reordered else raw).with_columns(
        pl.Series(sex_col, plink_sex(fp.sex)).cast(pl.String),
        # From the object array directly: a numpy unicode (astype(str)) column converts ~8x slower.
        pl.Series("sex_source", fp.sex_source, dtype=pl.String),
    )
    sets = fp.written_sets
    if sets is not None:
        written = pl.Series("multiple_birth_set", sets)
        written = pl.select(pl.when(written >= 0).then(written.cast(pl.String))).to_series()
        if fp.multiple_birth_set is None and "multiple_birth_set" in out.columns:
            _warn_replaced_sets(out["multiple_birth_set"], written, fp.cleared_multiple_birth, multiple_birth_col)
        # A --twin-max-days run replaces the column of an earlier validate.tsv.gz silently: it recomputes it.
        out = out.with_columns(written.alias("multiple_birth_set"))
    elif "multiple_birth_set" in out.columns:
        logger.warning(
            "dropped the input's multiple_birth_set column; pass --twin-max-days or --multiple-birth-col to write it"
        )
        out = out.drop("multiple_birth_set")
    for column, cleared in ((multiple_birth_col, fp.cleared_multiple_birth), (zygosity_col, fp.cleared_zygosity)):
        if column is not None and cleared is not None and cleared.any():
            out = out.with_columns(pl.when(pl.Series(cleared)).then(None).otherwise(pl.col(column)).alias(column))
    if missing_parent_tokens:
        tokens = list(missing_parent_tokens)
        out = out.with_columns(
            pl.when(pl.col(c).str.strip_chars().is_in(tokens)).then(None).otherwise(pl.col(c)).alias(c)
            for c in (mother_col, father_col)
        )
    if fp.filled_mother.any() or fp.filled_father.any():
        out = out.with_columns(
            pl.when(pl.Series(fp.filled_mother))
            .then(fp.labels.strings(fp.mothers))
            .otherwise(pl.col(mother_col))
            .alias(mother_col),
            pl.when(pl.Series(fp.filled_father))
            .then(fp.labels.strings(fp.fathers))
            .otherwise(pl.col(father_col))
            .alias(father_col),
        )
    if not fp.founders:
        return out
    n_new = len(fp.founders)
    # Filler cells are nulls (not "") so the CSV writer emits bare empty
    # fields; polars quotes explicit empty strings as "".
    data: dict[str, list[str | None]] = {col: [None] * n_new for col in out.columns}
    data[id_col] = [fp.labels.label(f["id"]) for f in fp.founders]
    data[sex_col] = plink_sex(np.array([f["sex"] for f in fp.founders], dtype=np.int8)).astype(str).tolist()
    data[mother_col] = ["-1"] * n_new
    data[father_col] = ["-1"] * n_new
    # A founder pedsum added has no recorded sex: it comes from the role it fills.
    data["sex_source"] = [SexSource.IMPUTED_FROM_MISSING.value] * n_new
    new_rows = pl.DataFrame(data, schema=dict.fromkeys(out.columns, pl.String))
    return pl.concat([new_rows, out.cast(pl.String)])


def multiple_birth_sets(mothers: np.ndarray, fathers: np.ndarray, birth_date: np.ndarray, max_days: int) -> np.ndarray:
    """Each row's multiple-birth set number (1, 2, ...), -1 for none.

    Within a Sibship (both parents known), siblings sorted by birth date join
    one set while each is born at most ``max_days`` after the one before, so
    triplets are one set. A row with an unknown (``NaT``) birth date is in no
    set. Sets are numbered by (mother, father, birth date). Same parents and
    date say nothing of zygosity: a set is not an MZ annotation. WARNs when a
    chain makes a set span more than ``max_days``.
    """
    sets = np.full(len(mothers), -1, dtype=np.int32)
    rows = np.flatnonzero((mothers != -1) & (fathers != -1) & ~np.isnat(birth_date))
    if len(rows) < 2:
        return sets
    days = birth_date.astype(np.int64)
    rows = rows[np.lexsort((days[rows], fathers[rows], mothers[rows]))]
    m, f, d = mothers[rows], fathers[rows], days[rows]
    linked = (m[1:] == m[:-1]) & (f[1:] == f[:-1]) & (d[1:] - d[:-1] <= max_days)
    starts = np.flatnonzero(np.concatenate(([True], ~linked)))
    group = np.cumsum(np.concatenate(([False], ~linked)))
    sizes = np.diff(np.append(starts, len(rows)))
    in_set = sizes[group] >= 2
    sets[rows[in_set]] = np.cumsum(sizes >= 2)[group[in_set]]
    # Each group is a run sorted by day: its span is its last day minus its first.
    span = d[starts + sizes - 1] - d[starts]
    wide = (sizes >= 2) & (span > max_days)
    if wide.any():
        logger.warning(
            "%d multiple-birth set(s) span more than --twin-max-days %d through a chain of births "
            "(widest %d days); check those birth dates",
            int(wide.sum()),
            max_days,
            int(span[wide].max()),
        )
    return sets


def _warn_replaced_sets(
    current: pl.Series, written: pl.Series, cleared: np.ndarray | None, source_col: str | None
) -> None:
    """WARN when the ``--multiple-birth-col`` sets replace an input ``multiple_birth_set`` holding other values.

    Rows the ``multiple_birth_sets`` Check cleared are findings already, so
    they are not counted.
    """
    differs = current.cast(pl.String).str.strip_chars().ne_missing(written)
    if cleared is not None:
        differs = differs & ~pl.Series(cleared)
    n = int(differs.sum())
    if not n:
        return
    if source_col == "multiple_birth_set":
        logger.warning("rewrote %d multiple_birth_set cell(s) as pedsum's set numbers (blank for none)", n)
    else:
        logger.warning(
            "replaced the input's multiple_birth_set column with the sets from --multiple-birth-col %s "
            "(%d row(s) differ)",
            source_col,
            n,
        )


def _keep_recorded_sex_source(derived: np.ndarray, raw: pl.DataFrame) -> np.ndarray:
    """``derived`` with the input's own ``sex_source`` where this run derived ``input``.

    Loading a Fixed Pedigree again finds every sex resolved, so without its
    column a sex ``validate`` filled or overrode would pass for a recorded one
    (ADR 0006). A repair this run made wins over the column. A column with
    values pedsum does not write is ignored with a WARNING.
    """
    if "sex_source" not in raw.columns:
        return derived
    tokens = raw["sex_source"].cast(pl.String).str.strip_chars()
    known = [s.value for s in SexSource]
    unknown = tokens.filter(tokens.is_not_null() & ~tokens.is_in(known)).unique().sort()
    if len(unknown):
        logger.warning(
            "input column sex_source has values pedsum does not write (%s); ignored", unknown.head(3).to_list()
        )
        return derived
    use = (derived == SexSource.INPUT.value) & tokens.is_not_null().to_numpy()
    if not use.any():
        return derived
    out = derived.copy()
    out[use] = tokens.to_numpy()[use]
    return out


def _log_sex_imputation(imp, *, allow_missing_sex: bool) -> None:
    sex = imp.imputed_sex
    n_overridden = int(imp.overridden_mask.sum())
    n_unresolved = int((sex == SEX_UNKNOWN).sum())
    if imp.n_imputed > 0 or n_overridden > 0 or n_unresolved > 0:
        logger.info(
            "sex imputation: %d from missing, %d overridden from role, "
            "%d unresolved (sex_source column has per-row detail)",
            imp.n_imputed,
            n_overridden,
            n_unresolved,
        )
    n_ambiguous = int(imp.ambiguous_mask.sum())
    if allow_missing_sex and n_ambiguous > 0:
        logger.info("kept %d row(s) with sex_role_ambiguity (--allow-missing-sex)", n_ambiguous)
    # Never a parent, so no role to sex them from (ADR 0006): kept with sex 0.
    n_unsexed = n_unresolved - n_ambiguous
    if n_unsexed > 0:
        logger.warning(
            "kept %d individual(s) with unknown sex who are never parents (sex 0); "
            "effective-size refuses its sex-dependent estimators while any row is unsexed",
            n_unsexed,
        )


def _build_added_founders(
    mothers: np.ndarray,
    fathers: np.ndarray,
    id_index: IdIndex,
    no_sex_check: bool,
    input_rows: Callable[[np.ndarray], np.ndarray] = lambda rows: rows,
) -> list[dict]:
    """Synthesize founder rows for missing parent IDs, sorted by ID; ``input_rows`` maps rows for the reasons."""
    moms_ref = unique_ints(mothers[mothers != -1])
    dads_ref = unique_ints(fathers[fathers != -1])
    moms_missing = set(moms_ref[id_index.get_indexer(moms_ref) == -1].tolist())
    dads_missing = set(dads_ref[id_index.get_indexer(dads_ref) == -1].tolist())
    conflicts = moms_missing & dads_missing
    moms_only = moms_missing - conflicts
    dads_only = dads_missing - conflicts

    out: list[dict] = []
    for mid in sorted(moms_only):
        rows = input_rows(np.where(mothers == mid)[0])
        out.append({"id": int(mid), "sex": SEX_FEMALE, "reason": f"referenced as mother in {rows_phrase(rows)}"})
    for did in sorted(dads_only):
        rows = input_rows(np.where(fathers == did)[0])
        out.append({"id": int(did), "sex": SEX_MALE, "reason": f"referenced as father in {rows_phrase(rows)}"})
    if no_sex_check:
        for cid in sorted(conflicts):
            rows_m = input_rows(np.where(mothers == cid)[0])
            rows_f = input_rows(np.where(fathers == cid)[0])
            out.append(
                {
                    "id": int(cid),
                    "sex": SEX_FEMALE,
                    "reason": (
                        f"--no-sex-check; conflicting roles "
                        f"(mother {rows_phrase(rows_m)}, father {rows_phrase(rows_f)})"
                    ),
                }
            )
    out.sort(key=lambda x: x["id"])
    return out


def next_free_id(ids: np.ndarray, mothers: np.ndarray, fathers: np.ndarray, tokens: tuple[str, ...] = ()) -> int:
    """One past the largest ID used anywhere in the pedigree, as a Python int so it cannot wrap.

    Also past every integer ``--missing-parent-token``: a phantom with a token's
    ID would read as a missing parent wherever the token is passed again.
    """
    token_ints = [int(t) for t in tokens if t.lstrip("+-").isdigit()]
    return max(int(ids.max()), int(mothers.max()), int(fathers.max()), *token_ints) + 1


def _build_phantom_parents(
    mothers: np.ndarray,
    fathers: np.ndarray,
    first_id: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Give each half-founder its own new founder in its missing parent slot.

    Returns ``(mothers, fathers, phantoms)``: the parent arrays with every
    half-founder's ``-1`` slot pointing at its phantom, and the phantom founder
    rows (female for a missing mother, male for a missing father). Phantoms are
    numbered from ``first_id`` in row order; pass :func:`next_free_id` of the
    input pedigree so they collide with no ID, including dropped ones, and
    sort after every existing or synthesized founder.

    A phantom is unrelated to everyone and not inbred, so kinship and F among
    the original individuals are unchanged. Founder contributions now sum to 1
    through every individual, at the cost of one more founder per half-founder.
    """
    rows = np.flatnonzero((mothers == -1) ^ (fathers == -1))
    pids = first_id + np.arange(len(rows), dtype=np.int64)
    for_mother = mothers[rows] == -1
    mothers = mothers.copy()
    fathers = fathers.copy()
    mothers[rows[for_mother]] = pids[for_mother]
    fathers[rows[~for_mother]] = pids[~for_mother]
    phantoms = [
        {"id": pid, "sex": SEX_FEMALE if m else SEX_MALE}
        for pid, m in zip(pids.tolist(), for_mother.tolist(), strict=True)
    ]
    return mothers, fathers, phantoms
