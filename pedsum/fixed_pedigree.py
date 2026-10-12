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
- sex resolved from parent role, recorded per row in ``sex_source``;
- rows in topological order (parents before children).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, logger
from pedsum.checks import rows_phrase
from pedsum.parse import plink_sex
from pedsum.pedigree_ops import unique_ints

if TYPE_CHECKING:
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
    ped_depth: np.ndarray  # int32 topological depth, founders 0
    input_row: np.ndarray  # each row's position in the input
    reordered: bool  # False when the input was already parents-first (input_row is the identity)
    filled_mother: np.ndarray  # bool: the mother slot holds a phantom parent
    filled_father: np.ndarray
    # Synthesized founder rows ({"id", "sex"[, "reason"]}): missing parents by ID, then phantoms.
    founders: list[dict]
    labels: IdLabels

    def frame(self) -> pl.DataFrame:
        """The input rows as ``load_and_validate`` returns them: ID codes, int8 sex, ``ped_depth``."""
        columns = {
            "id": self.ids,
            "sex": self.sex,
            "mother": self.mothers,
            "father": self.fathers,
            "sex_source": self.sex_source,
        }
        if self.birth_year is not None:
            columns["birth_year"] = self.birth_year
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
    founders = (
        _build_added_founders(mothers, fathers, ctx.id_index, ctx.config.no_sex_check) if synthesize_founders else []
    )
    filled_mother = filled_father = np.zeros(n, dtype=bool)
    if fill_half_founders:
        if next_id is None:
            next_id = next_free_id(ids, mothers, fathers)
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
        sex_source=take(imp.sex_source),
        birth_year=None if ctx.birth_year is None else take(ctx.birth_year),
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
) -> pl.DataFrame:
    """The string frame ``validate`` writes: the input's columns and tokens, with the fixes.

    ``raw`` is the input as read (all String columns). Each input row keeps
    its own tokens except the sex column, written in PLINK's coding
    (1=male, 2=female, 0=unknown) whatever tokens the input used, a parent
    written as one of ``missing_parent_tokens``, written blank so the file
    reads the same without ``--missing-parent-token``, and a parent slot that
    now holds a phantom. A ``sex_source`` column records which rows pedsum
    changed. Founder rows come first, with empty cells outside the ID, sex
    and parent columns.
    """
    out = (raw[fp.input_row] if fp.reordered else raw).with_columns(
        pl.Series(sex_col, plink_sex(fp.sex)).cast(pl.String),
        # From the object array directly: a numpy unicode (astype(str)) column converts ~8x slower.
        pl.Series("sex_source", fp.sex_source, dtype=pl.String),
    )
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
    new_rows = pl.DataFrame(data, schema=dict.fromkeys(out.columns, pl.String))
    return pl.concat([new_rows, out.cast(pl.String)])


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
    if allow_missing_sex:
        n_ambiguous = int(imp.ambiguous_mask.sum())
        if n_ambiguous > 0:
            logger.info("kept %d row(s) with sex_role_ambiguity (--allow-missing-sex)", n_ambiguous)
        n_orphan = int(((sex == SEX_UNKNOWN) & ~imp.ambiguous_mask).sum())
        if n_orphan > 0:
            logger.info("kept %d row(s) with sex=%d (--allow-missing-sex)", n_orphan, SEX_UNKNOWN)


def _build_added_founders(
    mothers: np.ndarray,
    fathers: np.ndarray,
    id_index: IdIndex,
    no_sex_check: bool,
) -> list[dict]:
    """Synthesize founder rows for missing parent IDs, sorted by ID."""
    moms_ref = unique_ints(mothers[mothers != -1])
    dads_ref = unique_ints(fathers[fathers != -1])
    moms_missing = set(moms_ref[id_index.get_indexer(moms_ref) == -1].tolist())
    dads_missing = set(dads_ref[id_index.get_indexer(dads_ref) == -1].tolist())
    conflicts = moms_missing & dads_missing
    moms_only = moms_missing - conflicts
    dads_only = dads_missing - conflicts

    out: list[dict] = []
    for mid in sorted(moms_only):
        rows = np.where(mothers == mid)[0]
        out.append({"id": int(mid), "sex": SEX_FEMALE, "reason": f"referenced as mother in {rows_phrase(rows)}"})
    for did in sorted(dads_only):
        rows = np.where(fathers == did)[0]
        out.append({"id": int(did), "sex": SEX_MALE, "reason": f"referenced as father in {rows_phrase(rows)}"})
    if no_sex_check:
        for cid in sorted(conflicts):
            rows_m = np.where(mothers == cid)[0]
            rows_f = np.where(fathers == cid)[0]
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


def next_free_id(ids: np.ndarray, mothers: np.ndarray, fathers: np.ndarray) -> int:
    """One past the largest ID used anywhere in the pedigree, as a Python int so it cannot wrap."""
    return max(int(ids.max()), int(mothers.max()), int(fathers.max())) + 1


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
