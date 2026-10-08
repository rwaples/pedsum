"""Mate Correlation over Mating Pairs: trait typing and strata here; the estimators and inference in pg-phenotype.

pedsum types the trait columns, labels each row's stratum and writes the
payload; ``pg_phenotype.assortative.mate_correlation`` computes the estimates,
their SEs and CIs, the Mate Network bootstrap and the permutation test
(pedsum#13's method, ported).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from pg_phenotype import PgPhenotypeError
from pg_phenotype import Trait as PgTrait
from pg_phenotype.assortative import mate_correlation

from pedsum.base import MIN_STRATUM_NETWORKS, PedigreeError

if TYPE_CHECKING:
    from pg_phenotype.assortative import Draws, EstimatorResult, Permutation

    from pedsum.base import TraitKind

StratifyBy = Literal["depth", "birth_year"]

#: More distinct numeric values than this makes an untyped trait continuous.
MAX_INFERRED_LEVELS = 20

_BINARY_WORDS: tuple[tuple[str, str], ...] = (("false", "true"), ("no", "yes"))

#: The YAML key each estimator's value is written under.
VALUE_KEYS = {
    "pearson": "r", "spearman": "r", "phi": "r", "point_biserial": "r", "odds_ratio": "value",
    "tetrachoric": "rho", "polychoric": "rho", "biserial": "rho", "polyserial": "rho",
}  # fmt: skip

NOTES = [
    "Mating Pairs are observed only through offspring; partnerships without a recorded child are absent.",
    "Phenotypic correlation; tetrachoric/polychoric/polyserial assume bivariate-normal liability, "
    "the other estimators describe the observed values.",
    "Censored age-dependent diagnoses are not corrected; a mate who has not yet been diagnosed counts as unaffected.",
]


@dataclass(frozen=True)
class Trait:
    """One typed trait column, aligned to the pedigree rows.

    ``values`` is float64 per row with NaN for missing; a binary or ordinal
    trait is coded ``0..k-1`` in the order of ``levels``.
    """

    name: str
    kind: TraitKind
    type_source: Literal["inferred", "stated"]
    levels: tuple[str, ...] | None
    values: np.ndarray

    @property
    def n_missing(self) -> int:
        """Rows without a value."""
        return int(np.isnan(self.values).sum())


def _level_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def classify_trait(name: str, tokens: np.ndarray, stated: TraitKind | None = None) -> Trait:
    """Type the raw tokens of one trait column (``None`` = missing) and code its values.

    Two levels infer binary, more than ``MAX_INFERRED_LEVELS`` numeric values
    infer continuous, and anything between needs ``stated``. Binary levels are
    two numbers (the lower is the reference) or ``false/true`` / ``no/yes`` in
    any case; ordinal levels are numeric and ordered numerically. Raises
    ``PedigreeError`` on an all-missing, constant, non-finite, non-numeric or
    ambiguous column.
    """
    source: Literal["inferred", "stated"] = "inferred" if stated is None else "stated"
    column = pl.Series(name, tokens.tolist(), dtype=pl.String)
    present = column.is_not_null().to_numpy()
    if not present.any():
        raise PedigreeError(f"trait {name!r} is missing in every row")
    observed = column.drop_nulls()
    numeric = observed.cast(pl.Float64, strict=False)
    if numeric.is_null().any():
        words = sorted(set(observed.str.to_lowercase().to_list()))
        if stated in (None, "binary") and tuple(words) in _BINARY_WORDS:
            values = np.full(len(tokens), np.nan)
            values[present] = (observed.str.to_lowercase() == words[1]).cast(pl.Float64).to_numpy()
            return Trait(name, "binary", source, (words[0], words[1]), values)
        if len(words) == 1:
            raise PedigreeError(f"trait {name!r} is constant ({words[0]!r} in every non-missing row)")
        samples = observed.filter(numeric.is_null()).unique(maintain_order=True).head(3).to_list()
        raise PedigreeError(
            f"trait {name!r} must be numeric (binary traits may also use true/false or yes/no); got {samples}"
        )
    values = numeric.to_numpy()
    if not np.isfinite(values).all():
        samples = observed.filter(~numeric.is_finite()).unique(maintain_order=True).head(3).to_list()
        raise PedigreeError(f"trait {name!r} holds non-finite value(s) {samples}")
    n_levels = len(np.unique(values))
    if n_levels == 1:
        raise PedigreeError(f"trait {name!r} is constant ({_level_label(float(values[0]))} in every non-missing row)")
    kind = stated
    if kind is None:
        if n_levels == 2:
            kind = "binary"
        elif n_levels > MAX_INFERRED_LEVELS:
            kind = "continuous"
        else:
            raise PedigreeError(
                f"trait {name!r} has {n_levels} distinct numeric values; pass "
                f"--trait-type {name}=ordinal or --trait-type {name}=continuous"
            )
    if kind == "binary" and n_levels != 2:
        raise PedigreeError(f"trait {name!r} is stated binary but has {n_levels} distinct values")
    out = np.full(len(tokens), np.nan)
    if kind == "continuous":
        out[present] = values
        return Trait(name, kind, source, None, out)
    levels, codes = np.unique(values, return_inverse=True)
    out[present] = codes
    return Trait(name, kind, source, tuple(_level_label(float(v)) for v in levels), out)


def strata(df: pl.DataFrame, by: StratifyBy, birth_year_bin: int) -> np.ndarray:
    """Stratum of each row: its Depth, or the first year of its birth-year bin; -1 when unknown."""
    if by == "depth":
        return df["ped_depth"].to_numpy().astype(np.int64)
    year = df["birth_year"].to_numpy().astype(np.int64)
    return np.where(year == -1, -1, year // birth_year_bin * birth_year_bin)


def _trait_record(trait: Trait) -> dict:
    record: dict = {"name": trait.name, "type": trait.kind, "type_source": trait.type_source}
    if trait.levels is not None:
        record["levels"] = list(trait.levels)
    n_missing = trait.n_missing
    record |= {"n_values": len(trait.values) - n_missing, "n_missing": n_missing}
    return record


def _draws(d: Draws) -> dict:
    return {"requested": d.requested, "valid": d.valid, "failed": d.failed, "failure_reasons": dict(d.failure_reasons)}


def _permutation_record(p: Permutation) -> dict:
    return {
        "p_perm": p.p,
        "p_perm_unavailable_reason": p.p_unavailable_reason,
        "permutation_statistic": p.statistic,
        "permutations": _draws(p.draws)
        | {
            "seed": p.seed,
            "n_fixed_fathers": p.n_fixed_fathers,
            "stopped_early": p.stopped_early,
            "draws_used": p.draws_used,
            "sequential_h": p.sequential_h,
        },
    }


def _estimator_record(r: EstimatorResult, stratum_counts: dict | None = None) -> dict:
    """One estimator's record; a stratified one's ``stratum_counts`` come before its permutation test."""
    key = VALUE_KEYS[r.estimator]
    stratum_counts = stratum_counts or {}
    if r.reason is not None:
        return {key: None, "reason": r.reason} | stratum_counts
    record: dict = {key: r.value}
    if r.boundary is not None:
        record["boundary"] = r.boundary
    record |= {
        "se": r.se,
        "ci": None if r.ci is None else list(r.ci),
        "ci_method": r.ci_method,
        "ci_unavailable_reason": r.ci_unavailable_reason,
    }
    if r.bootstrap is not None:
        record["bootstrap"] = _draws(r.bootstrap)
    record |= stratum_counts
    if r.permutation is not None:
        record |= _permutation_record(r.permutation)
    return record


def compute_assortative_mating(
    df: pl.DataFrame,
    traits: list[Trait],
    *,
    permutations: int,
    bootstrap: int,
    seed: int,
    stratify_by: StratifyBy | None = None,
    birth_year_bin: int = 10,
    min_stratum_networks: int = MIN_STRATUM_NETWORKS,
) -> dict:
    """The ``assortative_mating`` payload: Mate Correlation cells over the analysed Mating Pairs.

    With ``stratify_by``, pairs with a mate of unknown stratum are dropped
    before any cell, and each cell then drops the pairs in its strata that span
    fewer than ``min_stratum_networks`` Mate Networks or are degenerate, so a
    cell's crude and stratified estimates share one sample.
    ``seed`` keys every permutation and bootstrap draw.  pg-phenotype's
    refusals raise ``PedigreeError``.
    """
    pedigree = {"id": df["id"].to_numpy(), "mother": df["mother"].to_numpy(), "father": df["father"].to_numpy()}
    stratum = None
    if stratify_by is not None:
        labels = strata(df, stratify_by, birth_year_bin)
        stratum = np.where(labels == -1, np.nan, labels.astype(np.float64))
    try:
        pg_traits = [PgTrait(t.values, t.kind) for t in traits]
        result = mate_correlation(
            pedigree,
            pg_traits,
            stratum=stratum,
            permutations=permutations,
            bootstrap=bootstrap,
            seed=seed,
            min_stratum_networks=min_stratum_networks,
        )
    except PgPhenotypeError as e:
        raise PedigreeError(f"assortative mating: {e}") from e

    s = result.sample
    payload: dict = {
        "traits": [_trait_record(t) for t in traits],
        "settings": {
            "permutations": permutations,
            "bootstrap": bootstrap,
            "seed": seed,
            "threads": result.settings["threads"],
            "ci_level": result.settings["ci_level"],
            "ci_method": "bootstrap" if bootstrap > 0 else "sandwich",
            "stratify_by": stratify_by,
            "birth_year_bin": birth_year_bin if stratify_by == "birth_year" else None,
            "min_stratum_networks": min_stratum_networks if stratify_by is not None else None,
        },
        "inference": dict(result.method),
        "mating_pairs": {
            "n_total": s.n_total,
            "n_dropped": {"unknown_stratum": s.n_dropped_unknown_stratum},
            "n_mate_networks": s.n_mate_networks,
            "largest_mate_network_share": s.largest_mate_network_share,
            "n_mothers_multiple_mates": s.n_mothers_multiple_mates,
            "n_fathers_multiple_mates": s.n_fathers_multiple_mates,
        },
    }
    records = []
    for c in result.cells:
        d = c.n_dropped
        crude: dict = {} if c.table is None else {"table": [list(row) for row in c.table]}
        crude |= {r.estimator: _estimator_record(r) for r in c.crude}
        record: dict = {
            "mother": traits[c.mother_trait].name,
            "father": traits[c.father_trait].name,
            "n": c.n,
            "n_dropped": {
                "mother_missing": d.mother_missing,
                "father_missing": d.father_missing,
                "both_missing": d.both_missing,
                "small_stratum": d.small_stratum,
                "degenerate_stratum": d.degenerate_stratum,
            },
            "n_mate_networks": c.n_mate_networks,
            "largest_mate_network_share": c.largest_mate_network_share,
            "crude": crude,
        }
        if c.stratified is not None:
            st = c.stratified
            counts = {"n_strata_mothers": st.n_strata_mothers, "n_strata_fathers": st.n_strata_fathers}
            record["stratified"] = {st.result.estimator: _estimator_record(st.result, counts)}
        records.append(record)
    payload["mate_correlation"] = records
    if result.within_person is not None:
        payload["within_person"] = {}
        for sex, w in result.within_person.items():
            key = VALUE_KEYS[w.estimator]
            wp: dict = {"estimator": w.estimator, "n": w.n}
            if w.reason is not None:
                wp |= {key: None, "reason": w.reason}
            else:
                wp[key] = w.value
                if w.boundary is not None:
                    wp["boundary"] = w.boundary
            payload["within_person"][sex] = wp
    payload["notes"] = list(NOTES)
    return payload
