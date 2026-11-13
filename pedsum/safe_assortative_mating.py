"""``assortative-mating --safe-attempt``: the release list, and the payload it is applied to.

Safe mode writes nothing the ordinary ``assortative_mating.yaml`` payload
holds. :func:`build_source` builds a separate *source* payload: the couples
each number rests on, counted again from the pedigree, and the estimators'
records read field by field. :func:`pedsum.disclosure.build_release` copies
from it only the leaves :func:`assortative_mating_spec` lists, and
:func:`safe_payload` arranges that release, and nothing else, into the file.

The supporting unit of a count of Mating Pairs is the couple, held to the
people behind it: a count of couples is backed by the smaller of its
distinct mothers and its distinct fathers, so three couples of one man are
one unit, not three.

pg-phenotype reports how many couples a cell analyses, not which. The
support of a cell is counted on the couples themselves, so
:func:`build_source` selects them again by pg-phenotype's rules (complete
values, then the thin- and degenerate-stratum fixed point) and refuses to
write when its counts differ from the ones pg-phenotype reported.

One table is released that the ordinary file lacks: the couples by level of
a cell with an ordinal trait. The polychoric correlation is fitted to it, so
it is the table that gates that estimate, as the 2x2 gates the tetrachoric.

What the list leaves out, and why:

* Every extremum: the largest Mate Network's share, of the sample and of
  each cell.
* Every count that is the complement of a released one: the couples a cell
  drops for a missing value or a thin stratum, the couples of unknown
  stratum, the sample's Mate Network count beside each cell's.
* The levels of a numeric trait, which are data values, and the reasons of
  failed resampling draws, whose names the list cannot enumerate.

The counts of couples are linked: the couples total and each cell's ``n``
are sums over the same couples, grouped by which cells analyse them. Those
groups are a gate. One small group withholds every ``n``, and with them
every estimate: with two traits the four ``n`` and the total give the small
group back by inclusion and exclusion.

Three more counts are sums over those groups, and each is gated on the
parts a reader could subtract out:

* A cell's Mate Networks are its couples less the couples beyond one per
  network. They are withheld with the cell's ``n``, and when the couples
  beyond one per network differ, by fewer than the minimum cell, from
  another cell's or from the extra mates of the whole sample.
* Two tables with the same mothers' (or fathers') trait differ in that
  margin by the levels of the couples one cell analyses and the other does
  not. Grouped by which of those cells analyse them, a level of a group
  held by fewer than the minimum cell withholds every table with that
  margin.
* The individuals with and without a trait are the mothers, the fathers
  and everyone else. The cells' counts give the parents' share, so the
  table is withheld when any of the three parts is small.

The individuals are themselves a table of three parts with a released
total: the couples' mothers, their fathers and everyone else. A pedigree in
which fewer than the minimum cell are in no couple withholds the three
counts that say how many distinct mothers and fathers there are: the
parents with several mates, the parents with both traits, and each cell's
Mate Networks. The counts of couples stay: without those three they bound
the mothers and the fathers from above only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from pedsum.assortative_mating import VALUE_KEYS, strata
from pedsum.base import PedigreeError
from pedsum.disclosure import (
    Contingency,
    Count,
    Derived,
    Link,
    Plain,
    ReleaseSpec,
    Table,
    build_release,
    shared_total,
)
from pedsum.pedigree_ops import IdIndex, rows_of_parents

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl

    from pedsum.assortative_mating import StratifyBy, Trait
    from pedsum.disclosure import Path, Rule

_META = ("input", "command", "version", "generated_at")
_N: Path = ("n_total",)
_SECTION = "assortative_mating"

_TRAIT_FIELDS = ("name", "type", "type_source", "levels")
_TRAIT_ROWS: Path = ("supports", "trait_rows")
_ROWS_BY_ROLE: Path = ("gates", "trait_rows_by_role")
_ROWS_SUPPORT: Path = ("gates", "trait_rows_support")
_ROLES: Path = ("gates", "individuals_by_role")
_GROUP_LEVELS: Path = ("gates", "couples_by_cells_levels")
_STATES = ("with_value", "missing")
_PAIRS: Path = ("mating_pairs", "n_total")
_PAIRS_FLOOR: Path = ("gates", "mating_pairs_floor")
_GROUPS: Path = ("gates", "couples_by_cells")
_SEXES = {"mothers": "mother", "fathers": "father"}

#: The levels of a binary trait written as words, the only levels that are not numbers read from the data.
WORD_LEVELS = (("false", "true"), ("no", "yes"))

#: The ``reason`` of an estimate that was withheld. A constant, not data.
WITHHELD = "withheld"

_SETTINGS = {
    "permutations": "The --permutations option of the run.",
    "bootstrap": "The --bootstrap option of the run.",
    "seed": "The --seed option of the run, which keys the resampling draws and is not data.",
    "threads": "The number of threads the run used, a property of the machine and the --threads option.",
    "ci_level": "The confidence level of every interval, the same for every pedigree.",
    "ci_method": "Whether the intervals are bootstrap or sandwich intervals, set by --bootstrap.",
    "stratify_by": "The --stratify-by option of the run.",
    "birth_year_bin": "The --birth-year-bin option of the run, the width of a birth-year stratum.",
    "min_stratum_networks": "The --min-stratum-networks option of the run.",
}

#: The keys of pg-phenotype's method description, each a fixed text.
_INFERENCE = (
    *("se_method", "ci_scale", "bootstrap_unit", "bootstrap_assumption", "bootstrap_method"),
    *("permutation_null", "permutation_blocks", "permutation_statistic", "permutation_stopping"),
    "permutation_stop_h",
)

# The fields of an estimator's record that are copied, in the ordinary file's order. ``failure_reasons`` is a
# mapping keyed by pg-phenotype's reason names, which a fixed list cannot enumerate, and ``n_fixed_fathers`` is a
# count of fathers, released once per cell under the minimum-cell rule.
_ESTIMATE_FIELDS = ("boundary", "se", "ci", "ci_method", "ci_unavailable_reason")
_DRAWS_FIELDS = ("requested", "valid", "failed")
_TEST_FIELDS = ("p_perm", "p_perm_unavailable_reason", "permutation_statistic")
_PERMUTATION_FIELDS = (*_DRAWS_FIELDS, "seed", "stopped_early", "draws_used", "sequential_h")


@dataclass(frozen=True)
class _Estimator:
    """One estimator: the table of couples it is computed from (None for raw values), and why it may be released."""

    gate: str | None
    reason: str


_ON_VALUES = (
    "computed from the two mates' trait values over the cell's analysed couples, released only when those "
    "couples come from at least the minimum cell of distinct mothers, of distinct fathers and of Mate "
    "Networks; no value, mean or extreme of the trait is released beside it."
)
_ON_TABLE = (
    "of the cell's table of couples by the two mates' levels, released only when that table is, so the point "
    "estimate adds no number to it; its standard error, interval and permutation p-value describe how those "
    "couples group into Mate Networks."
)
_ON_LEVELS = (
    "computed from one mate's trait values within each level of the other's, released only when every level "
    "is held by no analysed couple or by at least the minimum cell of distinct mothers and of distinct fathers."
)

_ESTIMATORS: dict[str, _Estimator] = {
    "pearson": _Estimator(None, f"Pearson correlation, {_ON_VALUES}"),
    "spearman": _Estimator(None, f"Spearman rank correlation, {_ON_VALUES}"),
    "tetrachoric": _Estimator("table", f"Tetrachoric correlation, the maximum-likelihood fit {_ON_TABLE}"),
    "odds_ratio": _Estimator("table", f"Odds ratio, the cross-product ratio {_ON_TABLE}"),
    "phi": _Estimator("table", f"Phi coefficient, the Pearson correlation {_ON_TABLE}"),
    "polychoric": _Estimator("table", f"Polychoric correlation, the maximum-likelihood fit {_ON_TABLE}"),
    "biserial": _Estimator("levels", f"Biserial correlation, {_ON_LEVELS}"),
    "point_biserial": _Estimator("levels", f"Point-biserial correlation, {_ON_LEVELS}"),
    "polyserial": _Estimator("levels", f"Polyserial correlation, {_ON_LEVELS}"),
}
assert set(_ESTIMATORS) == set(VALUE_KEYS)


# ---------------------------------------------------------------------------
# The couples, counted again
# ---------------------------------------------------------------------------


def _labelled(values: Sequence[int] | np.ndarray, labels: Sequence[Any] | None = None) -> dict[str, int]:
    values = np.asarray(values).tolist()
    return {str(label): value for label, value in zip(labels or range(len(values)), values, strict=True)}


@dataclass(frozen=True)
class _Couples:
    """Mating Pairs as rows: couple ``p`` is ``(mothers[p], fathers[p])``, over a pedigree of ``n_rows``."""

    mothers: np.ndarray
    fathers: np.ndarray
    n_rows: int

    def __len__(self) -> int:
        return self.mothers.size

    def take(self, index: np.ndarray) -> _Couples:
        return _Couples(self.mothers[index], self.fathers[index], self.n_rows)

    def _distinct(self, group: np.ndarray, person: np.ndarray, n_groups: int) -> np.ndarray:
        keys = np.unique(group.astype(np.int64) * self.n_rows + person)
        return np.bincount(keys // self.n_rows, minlength=n_groups)

    def support(self, group: np.ndarray | None = None, n_groups: int = 1) -> np.ndarray:
        """Per group of couples, the smaller of its distinct mothers and its distinct fathers."""
        if group is None:
            group = np.zeros(len(self), dtype=np.int64)
        return np.minimum(
            self._distinct(group, self.mothers, n_groups), self._distinct(group, self.fathers, n_groups)
        ).astype(np.int64)

    def network_labels(self) -> tuple[np.ndarray, int]:
        """Each couple's Mate Network, and the number of rows a label is below."""
        shape = (self.n_rows, self.n_rows)
        graph = coo_array((np.ones(len(self), dtype=bool), (self.mothers, self.fathers)), shape=shape)
        _, labels = connected_components(graph, directed=False)
        return labels[self.mothers], self.n_rows

    def n_networks(self) -> int:
        labels, _ = self.network_labels()
        return int(np.unique(labels).size)


def _mating_pairs(df: pl.DataFrame) -> _Couples:
    """The distinct (mother, father) of the children whose parents are both rows, as pg-phenotype finds them."""
    index = IdIndex(df["id"].to_numpy())
    mother = rows_of_parents(df["mother"].to_numpy(), index)
    father = rows_of_parents(df["father"].to_numpy(), index)
    both = (mother >= 0) & (father >= 0)
    n = len(df)
    keys = np.unique(mother[both] * n + father[both])
    return _Couples(keys // n, keys % n, n)


def _dense(labels: np.ndarray) -> tuple[np.ndarray, int]:
    values, codes = np.unique(labels, return_inverse=True)
    return codes, values.size


def _thin_or_constant(
    stratum: np.ndarray, values: np.ndarray, networks: np.ndarray, n_labels: int, min_networks: int
) -> np.ndarray:
    """The couples whose stratum spans fewer than ``min_networks`` Mate Networks or holds one trait value."""
    code, k = _dense(stratum)
    spanned = np.bincount(np.unique(code * n_labels + networks) // n_labels, minlength=k)
    lo, hi = np.full(k, np.inf), np.full(k, -np.inf)
    np.minimum.at(lo, code, values)
    np.maximum.at(hi, code, values)
    return (spanned[code] < min_networks) | (lo[code] == hi[code])


def _analysed(
    couples: _Couples,
    x: np.ndarray,
    y: np.ndarray,
    stratum: np.ndarray | None,
    min_networks: int,
) -> np.ndarray:
    """The couples one cell analyses: both values present, then pg-phenotype's thin-stratum fixed point."""
    keep = np.flatnonzero(~np.isnan(x) & ~np.isnan(y))
    while stratum is not None and keep.size:
        kept = couples.take(keep)
        networks, n_labels = kept.network_labels()
        bad = _thin_or_constant(stratum[kept.mothers], x[keep], networks, n_labels, min_networks)
        bad |= _thin_or_constant(stratum[kept.fathers], y[keep], networks, n_labels, min_networks)
        if not bad.any():
            break
        keep = keep[~bad]
    return keep


def _floor(count: int, *parts: int | np.ndarray) -> int:
    """The smallest nonzero part: the support of a count released only when each part is empty or large."""
    nonzero = [int(v) for part in parts for v in np.atleast_1d(part) if v > 0]
    return min([count, *nonzero])


def _level_table(
    couples: _Couples, rows: np.ndarray, n_rows: int, cols: np.ndarray, n_cols: int
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Couples by two codes, and each cell's support, as nested mappings keyed by code."""
    group = rows * n_cols + cols
    counts = np.bincount(group, minlength=n_rows * n_cols).reshape(n_rows, n_cols)
    support = couples.support(group, n_rows * n_cols).reshape(n_rows, n_cols)
    return (
        {str(r): _labelled(counts[r]) for r in range(n_rows)},
        {str(r): _labelled(support[r]) for r in range(n_rows)},
    )


def _n_levels(trait: Trait) -> int | None:
    return None if trait.levels is None else len(trait.levels)


def _mismatch(what: str, counted: object, reported: object) -> PedigreeError:
    return PedigreeError(
        f"safe-attempt: counted {what} as {counted!r} where pg-phenotype reports {reported!r}; no file is "
        "written, because the supports would not be those of the reported estimates",
        public=f"safe-attempt: {what} counted here differs from what pg-phenotype reports; no file is written, "
        "because the supports would not be those of the reported estimates",
    )


def _draws(record: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {key: record[key] for key in fields if key in record}


def _estimate(name: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """The listed fields of one estimator's record."""
    key = VALUE_KEYS[name]
    out: dict[str, Any] = {key: record.get(key)}
    if record.get("reason") is not None:
        return out | {"reason": record["reason"]}
    out |= _draws(record, _ESTIMATE_FIELDS)
    if "bootstrap" in record:
        out["bootstrap"] = _draws(record["bootstrap"], _DRAWS_FIELDS)
    out |= _draws(record, _TEST_FIELDS)
    if "permutations" in record:
        out["permutations"] = _draws(record["permutations"], _PERMUTATION_FIELDS)
    return out


def _estimates(records: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _estimate(name, record) for name, record in records.items() if name in _ESTIMATORS}


def _cell_source(
    key: str,
    record: Mapping[str, Any],
    couples: _Couples,
    mother: Trait,
    father: Trait,
    stratum: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One cell's released leaves and its gates, from pg-phenotype's ``record`` and the cell's analysed ``couples``."""
    x, y = mother.values[couples.mothers], father.values[couples.fathers]
    n = len(couples)
    if n != record["n"]:
        raise _mismatch(f"the analysed couples of cell {key}", n, record["n"])
    cell: dict[str, Any] = {"mother": mother.name, "father": father.name, "n": n}
    gates: dict[str, Any] = {}
    if record.get("n_mate_networks") is not None:
        n_networks = couples.n_networks()
        if n_networks != record["n_mate_networks"]:
            raise _mismatch(f"the Mate Networks of cell {key}", n_networks, record["n_mate_networks"])
        cell["n_mate_networks"] = n_networks
        gates["networks"] = n_networks

    k_m, k_f = _n_levels(mother), _n_levels(father)
    if k_m is not None and k_f is not None:
        cell["table"], gates["table_support"] = _level_table(couples, x.astype(np.int64), k_m, y.astype(np.int64), k_f)
        reported = record["crude"].get("table")
        counted = [list(row.values()) for row in cell["table"].values()]
        if reported is not None and counted != reported:
            raise _mismatch(f"the table of cell {key}", counted, reported)
    elif k_m is not None or k_f is not None:
        codes, k = (x, k_m) if k_m is not None else (y, k_f)
        assert k is not None
        gates["levels"] = _labelled(couples.support(codes.astype(np.int64), k))

    cell["crude"] = _estimates(record["crude"])
    tests = [record["crude"], record.get("stratified", {})]
    fixed = [r["permutations"]["n_fixed_fathers"] for rs in tests for r in rs.values() if "permutations" in r]
    if fixed:
        cell["n_fixed_fathers"] = fixed[0]

    if stratum is not None and "stratified" in record:
        cell["stratified"] = _estimates(record["stratified"])
        (stratified,) = record["stratified"].values()
        for sex, person, codes, k in (("mother", couples.mothers, x, k_m), ("father", couples.fathers, y, k_f)):
            code, n_strata = _dense(stratum[person])
            if n_strata != stratified[f"n_strata_{sex}s"]:
                raise _mismatch(f"the {sex} strata of cell {key}", n_strata, stratified[f"n_strata_{sex}s"])
            gates[f"{sex}_strata"] = _labelled(couples.support(code, n_strata))
            if k is not None:
                # The thresholds of a discrete trait are fitted per stratum, from its levels there.
                _, gates[f"{sex}_stratum_levels"] = _level_table(couples, code, n_strata, codes.astype(np.int64), k)
        cell["n_strata"] = {"mothers": stratified["n_strata_mothers"], "fathers": stratified["n_strata_fathers"]}
    return cell, gates


def _within_person_source(
    sex: str, record: Mapping[str, Any], persons: np.ndarray, traits: Sequence[Trait], n_couples: int, floor: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One sex's within-person leaves and gates, over the distinct ``persons`` of the couples.

    ``floor`` is the smallest of the support of the couples total, the
    individuals in no couple, and the networks floor of each cell whose
    Mate Networks the distinct persons would give: where nobody remates the
    persons with both traits are a count of couples, and where no trait is
    missing they are every mother or every father.
    """
    first, second = (trait.values[persons] for trait in traits)
    both = ~np.isnan(first) & ~np.isnan(second)
    n = int(both.sum())
    if n != record["n"]:
        raise _mismatch(f"the {sex} with both traits", n, record["n"])
    name = record["estimator"]
    out = {"estimator": name, "n": n, "estimate": _estimate(name, record)}
    # A person is in one couple per mate, so the couples less the persons counts the rematings; beside the
    # couples total, one remating would be one person.
    gates: dict[str, Any] = {"n_floor": min(_floor(n, n_couples - n), floor)}
    k_first, k_second = (_n_levels(trait) for trait in traits)
    first, second = first[both], second[both]
    if k_first is not None and k_second is not None:
        counts = np.bincount(first.astype(np.int64) * k_second + second.astype(np.int64), minlength=k_first * k_second)
        gates["table"] = {str(r): _labelled(row) for r, row in enumerate(counts.reshape(k_first, k_second))}
    elif k_first is not None or k_second is not None:
        codes, k = (first, k_first) if k_first is not None else (second, k_second)
        assert k is not None
        gates["levels"] = _labelled(np.bincount(codes.astype(np.int64), minlength=k))
    return out, gates


def _rows_by_role(trait: Trait, persons: Mapping[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """Individuals with and without a value of ``trait``: the couples' mothers, their fathers, and everyone else."""
    missing = np.isnan(trait.values)
    role = np.full(missing.size, len(persons), dtype=np.int64)
    for code, rows in enumerate(persons.values()):
        role[rows] = code
    counts = np.bincount(role * 2 + missing, minlength=2 * (len(persons) + 1)).reshape(-1, 2)
    return {name: _labelled(row, _STATES) for name, row in zip((*persons, "others"), counts, strict=True)}


def _individuals_by_role(by_role: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """The individuals who are a mother of the couples, a father of them, or neither."""
    return {role: sum(cells.values()) for role, cells in by_role.items()}


def _rows_support(by_role: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """Per state, the smallest nonzero part of its individuals among the roles."""
    parts = {state: [cells[state] for cells in by_role.values()] for state in _STATES}
    return {state: _floor(sum(values), *values) for state, values in parts.items()}


def _group_levels(
    every: _Couples, analysed_by: np.ndarray, traits: Sequence[Trait]
) -> dict[str, dict[str, dict[str, int]]]:
    """Per sex and discrete trait, the support of each group of couples at each level of that mate's trait.

    The tables of the cells with one trait on one side share its margin, so
    the couples are grouped by which of those cells analyse them; the
    couples none of them analyses are in no table.
    """
    out = {}
    n_traits = len(traits)
    cell_bit = 1 << np.arange(n_traits * n_traits).reshape(n_traits, n_traits)
    for sex, persons, bits in (
        ("mother", every.mothers, cell_bit.sum(axis=1)),
        ("father", every.fathers, cell_bit.sum(axis=0)),
    ):
        for i, trait in enumerate(traits):
            k = _n_levels(trait)
            if k is None:
                continue
            shared = analysed_by & bits[i]
            keep = np.flatnonzero(shared)
            masks, group = np.unique(shared[keep], return_inverse=True)
            level = trait.values[persons[keep]].astype(np.int64)
            support = every.take(keep).support(group * k + level, masks.size * k).reshape(masks.size, k)
            out[f"{sex}_{i}"] = {str(mask): _labelled(row) for mask, row in zip(masks.tolist(), support, strict=True)}
    return out


def _margin_floor(levels: Mapping[str, Mapping[str, int]]) -> int | None:
    """The smallest nonzero level of a group, when the margin's cells tell at least two groups apart."""
    if len(levels) < 2:
        return None
    return min(value for row in levels.values() for value in row.values() if value > 0)


def build_source(
    *,
    meta: Mapping[str, Any],
    payload: Mapping[str, Any],
    df: pl.DataFrame,
    traits: Sequence[Trait],
    stratify_by: StratifyBy | None,
    birth_year_bin: int,
    min_stratum_networks: int,
) -> dict[str, Any]:
    """The payload the release list is applied to: the estimators' records, and the couples they rest on.

    ``payload`` is :func:`pedsum.assortative_mating.compute_assortative_mating`'s.
    Only the fields named in this module are read from it, so a field
    pg-phenotype adds is not in the source. The couples are found and
    selected again from ``df`` and ``traits``, in memory proportional to the
    couples.

    Raises:
        PedigreeError: A count of the selected couples differs from the one
            pg-phenotype reported.
    """
    every = _mating_pairs(df)
    pairs = payload["mating_pairs"]
    if len(every) != pairs["n_total"]:
        raise _mismatch("the Mating Pairs", len(every), pairs["n_total"])
    stratum = None
    known = np.arange(len(every))
    if stratify_by is not None:
        stratum = strata(df, stratify_by, birth_year_bin)
        known = np.flatnonzero((stratum[every.mothers] != -1) & (stratum[every.fathers] != -1))
    kept = every.take(known)

    cells: dict[str, Any] = {}
    cell_gates: dict[str, Any] = {}
    analysed_by = np.zeros(len(every), dtype=np.int64)
    records = iter(payload["mate_correlation"])
    for i, mother in enumerate(traits):
        for j, father in enumerate(traits):
            key = f"{i}x{j}"
            x, y = mother.values[kept.mothers], father.values[kept.fathers]
            index = _analysed(kept, x, y, stratum, min_stratum_networks)
            analysed_by[known[index]] |= 1 << len(cells)
            cells[key], cell_gates[key] = _cell_source(key, next(records), kept.take(index), mother, father, stratum)

    # Couples grouped by which cells analyse them: every released count of couples is a sum of these groups,
    # and a cell is backed by no fewer people than the smallest group in it.
    sets, group = np.unique(analysed_by, return_inverse=True)
    group_support = every.support(group, sets.size)
    group_levels = _group_levels(every, analysed_by, traits)
    persons = {sex: np.unique(rows) for sex, rows in (("mothers", kept.mothers), ("fathers", kept.fathers))}
    rows_by_role = {str(i): _rows_by_role(trait, persons) for i, trait in enumerate(traits)}
    roles = _individuals_by_role(rows_by_role["0"])
    # A person is in one couple per mate, so the couples less the persons are the mates beyond each person's first.
    extra_mates = sum(len(kept) - rows.size for rows in persons.values())
    beyond = {key: cell["n"] - cell["n_mate_networks"] for key, cell in cells.items() if "n_mate_networks" in cell}
    pinned = dict.fromkeys(persons, len(every))
    for (key, cell), (i, j) in zip(cells.items(), np.ndindex(len(traits), len(traits)), strict=True):
        gate = cell_gates[key]
        floor = _floor(cell["n"], group_support[sets != 0])
        gate["n_floor"] = floor
        if key in beyond:
            # The couples beyond one per network count rematings: a single one would be one person of the cell,
            # and a difference of one from another cell's would be one remating among the couples they differ by.
            gaps = [abs(beyond[key] - other) for other in (*beyond.values(), extra_mates)]
            # A network holds a mother and a father, so the networks of couples who never remate are both, and
            # the mothers or fathers of the sample beyond them are the ones no couple of the cell holds. Where
            # nobody remates at all those are couples of the other groups, which the cell's floor covers.
            outside = [rows.size - cell["n"] for rows in persons.values()] if extra_mates and not beyond[key] else []
            gate["networks_floor"] = min(
                floor, _floor(cell["n_mate_networks"], beyond[key], *gaps, roles["others"], *outside)
            )
            # Where every extra mate of the sample is in the cell, its networks are no fewer than the couples
            # less those mates, and no more than either sex: the distinct persons of a sex they equal give them.
            if beyond[key] == extra_mates:
                for sex, rows in persons.items():
                    if rows.size == cell["n_mate_networks"]:
                        pinned[sex] = min(pinned[sex], gate["networks_floor"])
        if "table" in cell:
            # A withheld n withholds the table's total, which is the same number. A small level of a group
            # withholds the table: its margin less another table's is that group's levels.
            margins = (_margin_floor(group_levels[f"mother_{i}"]), _margin_floor(group_levels[f"father_{j}"]))
            floor = min([floor, *(margin for margin in margins if margin is not None)])
            gate["table_support"] = {
                row: {col: min(support, floor) for col, support in cols.items()}
                for row, cols in gate["table_support"].items()
            }

    gates: dict[str, Any] = {
        "couples_by_cells": _labelled(np.bincount(group, minlength=sets.size), sets.tolist()),
        "couples_by_cells_support": _labelled(group_support, sets.tolist()),
        "couples_by_cells_levels": group_levels,
        "mating_pairs_floor": _floor(len(every), every.support(), group_support),
        "trait_rows_by_role": rows_by_role,
        "trait_rows_support": {index: _rows_support(by_role) for index, by_role in rows_by_role.items()},
        "individuals_by_role": roles,
        "cells": cell_gates,
    }
    mating_pairs: dict[str, Any] = {"n_total": len(every)}
    for sex, rows in (("mothers", kept.mothers), ("fathers", kept.fathers)):
        _, mates = np.unique(rows, return_counts=True)
        sizes, held = np.unique(mates, return_counts=True)
        gates[f"mates_per_{_SEXES[sex]}"] = _labelled(held, sizes.tolist())
        several = int((mates > 1).sum())
        if several != pairs[f"n_{sex}_multiple_mates"]:
            raise _mismatch(f"the {sex} with several mates", several, pairs[f"n_{sex}_multiple_mates"])
        mating_pairs[f"n_{sex}_multiple_mates"] = several

    source: dict[str, Any] = {
        **{key: meta[key] for key in (*_META, *_N)},
        "traits": {
            str(i): {"name": trait.name, "type": trait.kind, "type_source": trait.type_source}
            # The levels of a numeric trait are values of the data; the two words of a binary one are not.
            | ({"levels": list(trait.levels)} if trait.levels in WORD_LEVELS else {})
            for i, trait in enumerate(traits)
        },
        "settings": {key: payload["settings"][key] for key in _SETTINGS},
        "inference": {key: payload["inference"][key] for key in _INFERENCE if key in payload["inference"]},
        "supports": {
            "trait_rows": {
                str(i): {"with_value": len(trait.values) - trait.n_missing, "missing": trait.n_missing}
                for i, trait in enumerate(traits)
            }
        },
        "gates": gates,
        "mating_pairs": mating_pairs,
        "cells": cells,
        "notes": list(payload["notes"]),
    }
    if "within_person" in payload:
        source["within_person"], gates["within_person"] = {}, {}
        for sex, record in payload["within_person"].items():
            source["within_person"][sex], gates["within_person"][sex] = _within_person_source(
                sex,
                record,
                persons[sex],
                traits,
                len(kept),
                min(_floor(gates["mating_pairs_floor"], roles["others"]), pinned[sex]),
            )
    return source


# ---------------------------------------------------------------------------
# Release list
# ---------------------------------------------------------------------------

_GATE = "It decides whether estimates are released and is not written to the file."


def _cell_rules(key: str, cell: Mapping[str, Any], gates: Mapping[str, Any]) -> list[Rule]:
    at: Path = ("cells", key)
    gate: Path = ("gates", "cells", key)
    n: Path = (*at, "n")
    rules: list[Rule] = [
        Plain((*at, sex), f"The name of the {sex}s' trait column, part of the --trait option.")
        for sex in _SEXES.values()
    ]
    rules += [
        Count(
            n,
            "The couples the cell analyses, released only when they come from at least the minimum cell of "
            "distinct mothers and of distinct fathers, and every group of couples analysed by one set of "
            "cells and not another does too: the differences between the cells' counts are those groups.",
            support=(*gate, "n_floor"),
        ),
        Count(
            (*at, "n_mate_networks"),
            "The Mate Networks of the cell's couples, the clusters its standard errors are over; withheld "
            "with the cell's count of couples, which it equals when nobody remates, and when the couples "
            "exceed the networks by fewer than the minimum cell, or by that few more or fewer than in "
            "another cell or than the extra mates of the whole sample: those differences count the few "
            "people who remated, in the cell or among the couples two cells differ by. Withheld too when "
            "fewer than the minimum cell of individuals are in no couple: the networks of couples who never "
            "remate are their mothers and their fathers. For the same reason, withheld when the cell's "
            "couples never remate, someone else does, and fewer than the minimum cell of mothers, or of "
            "fathers, are in no couple of the cell.",
            support=(*gate, "networks_floor"),
        ),
        Count(
            (*at, "n_fixed_fathers"),
            "The cell's fathers that the permutation test never exchanges, alone in their stratum and "
            "missingness pattern; none, or at least the minimum cell.",
        ),
    ]
    networks: Path = (*gate, "networks")
    rules.append(
        Count(
            networks,
            "The Mate Networks of the cell's couples again, without the remating floor: the standard errors, "
            f"intervals and bootstrap draws are statistics over these clusters. {_GATE}",
        )
    )
    base: tuple[Path, ...] = (n, networks)
    sources: dict[str | None, tuple[Path, ...]] = {None: base}
    if "table" in cell:
        table: Path = (*at, "table")
        rules.append(
            Contingency(
                table,
                "Couples by the mother's and the father's level, each cell backed by the smaller of its "
                "distinct mothers and distinct fathers; released whole, and only when no cell is small, no "
                "empty cell is in a small row or column, and no group of couples that another table with "
                "the same mothers' or fathers' trait leaves out holds a level of it in fewer than the "
                "minimum cell.",
                support=(*gate, "table_support"),
            )
        )
        sources["table"] = (*base, table)
    if "levels" in gates:
        levels: Path = (*gate, "levels")
        rules.append(
            Table(
                levels,
                f"The smaller of the distinct mothers and distinct fathers of the couples at each level of "
                f"the cell's discrete trait. {_GATE}",
                ordered=False,
            )
        )
        sources["levels"] = (*base, levels)

    strata_gates: list[Path] = []
    for sex in _SEXES.values():
        if f"{sex}_strata" in gates:
            strata_gates.append((*gate, f"{sex}_strata"))
            rules.append(
                Table(
                    strata_gates[-1],
                    f"The smaller of the distinct mothers and distinct fathers of the couples in each {sex} "
                    f"stratum. {_GATE}",
                    ordered=False,
                )
            )
        if f"{sex}_stratum_levels" in gates:
            strata_gates.append((*gate, f"{sex}_stratum_levels"))
            rules.append(
                Contingency(
                    strata_gates[-1],
                    f"The same support by {sex} stratum and the level of the {sex}'s trait, the counts each "
                    f"stratum's thresholds are fitted to. {_GATE}",
                )
            )

    for form, extra in (("crude", ()), ("stratified", tuple(strata_gates))):
        for name, estimator in _ESTIMATORS.items():
            if name not in cell.get(form, {}):
                continue
            reason = estimator.reason
            if form == "stratified":
                reason = (
                    f"{reason} Its stratified form standardises each mate within their stratum, and is "
                    "withheld when any stratum of either sex is below the minimum cell."
                )
            rules.append(Derived((*at, form, name), reason, sources=(*sources[estimator.gate], *extra)))
    if strata_gates:
        rules.append(
            Derived(
                (*at, "n_strata"),
                "The number of strata the cell's mothers and its fathers span, a count of strata and not of "
                "people; released with the stratified estimate, when every stratum is supported.",
                sources=tuple(strata_gates),
            )
        )
    return rules


def _within_person_rules(sex: str, gates: Mapping[str, Any]) -> list[Rule]:
    at: Path = ("within_person", sex)
    gate: Path = ("gates", "within_person", sex)
    n: Path = (*at, "n")
    rules: list[Rule] = [
        Plain((*at, "estimator"), "The name of the estimator, set by the two traits' types."),
        Count(
            n,
            f"The distinct {sex} of the couples with a value of both traits, withheld when the couples exceed "
            "them by fewer than the minimum cell: beside the couples total that difference counts rematings. "
            "Withheld with the couples total too: where nobody remates it is a count of couples, a sum of "
            "the same groups. Withheld when fewer than the minimum cell of individuals are in no couple: "
            f"where no trait is missing it is every one of the {sex}. Withheld with the Mate Networks of a "
            f"cell that holds every extra mate and as many networks as there are {sex}: the couples and the "
            f"{sex} bound those networks from both sides.",
            support=(*gate, "n_floor"),
        ),
    ]
    sources: tuple[Path, ...] = (n,)
    if "table" in gates:
        rules.append(Contingency((*gate, "table"), f"The {sex} by their levels of the two traits. {_GATE}"))
        sources = (n, (*gate, "table"))
    elif "levels" in gates:
        rules.append(
            Table((*gate, "levels"), f"The {sex} by their level of the discrete trait. {_GATE}", ordered=False)
        )
        sources = (n, (*gate, "levels"))
    rules.append(
        Derived(
            (*at, "estimate"),
            f"The correlation of the two traits within the {sex}, each counted once, released only when at "
            "least the minimum cell of them have both traits and no level or pair of levels is held by fewer.",
            sources=sources,
        )
    )
    return rules


def assortative_mating_spec(source: Mapping[str, Any]) -> ReleaseSpec:
    """The release list for ``assortative-mating``, with the links among the released tables ``source`` holds.

    The traits, cells and estimators of a run set which leaves exist, so
    the rules are laid out per payload: one fixed set per trait, per cell
    and per sex, over the keys this module names. A key ``source`` holds
    that no rule names is not released.
    """
    rules: list[Rule] = [
        Plain((key,), "Describes the run, not the individuals: the command, its input path, version and time.")
        for key in _META
    ]
    rules.append(Count(_N, "The number of individuals in the pedigree, withheld when below the minimum cell."))
    trait_rows = []
    # The traits are numbered and a cell is a pair of them: no other key of the source names a rule.
    traits = [index for index in source["traits"] if index.isdecimal()]
    for index in traits:
        rules += [
            Plain(("traits", index, "name"), "The name of a trait column, part of the --trait option."),
            Plain(
                ("traits", index, "type"),
                "The trait's type, stated by --trait-type or inferred from whether the column holds two, "
                "a few or many distinct values.",
            ),
            Plain(("traits", index, "type_source"), "Whether the trait's type was stated or inferred."),
            Plain(
                ("traits", index, "levels"),
                "The two words of a binary trait written as words (false and true, or no and yes); the "
                "levels of a numeric trait are data values and are not released.",
            ),
            Table(
                (*_TRAIT_ROWS, index),
                "Individuals with and without a value of the trait; each is none or at least the minimum "
                "cell among the couples' mothers, among their fathers and among everyone else, or the "
                "table is withheld: the cells' counts say how many parents lack the trait.",
                ordered=False,
                support=(*_ROWS_SUPPORT, index),
            ),
            Contingency(
                (*_ROWS_BY_ROLE, index),
                f"The same individuals split into the couples' mothers, their fathers and everyone else. {_GATE}",
            ),
        ]
        trait_rows.append((*_TRAIT_ROWS, index))
    rules += [Plain(("settings", key), reason) for key, reason in _SETTINGS.items()]
    rules += [
        Plain(("inference", key), "Part of the method description, a fixed text that no pedigree changes.")
        for key in _INFERENCE
    ]
    rules.append(
        Count(
            _PAIRS,
            "The Mating Pairs of the pedigree, released only when they come from at least the minimum cell of "
            "distinct mothers and of distinct fathers and so does every group of couples the cells' counts "
            "tell apart, the couples no cell analyses among them.",
            support=_PAIRS_FLOOR,
        )
    )
    gates = source["gates"]
    rules.append(
        Table(
            _ROLES,
            "The individuals who are a mother of the couples, a father of them, or neither. A reader who "
            "knew the distinct mothers and fathers would have the third part from the number of individuals, "
            f"so a small part withholds the counts that give them. {_GATE}",
            ordered=False,
        )
    )
    rules.append(
        Table(
            _GROUPS,
            "Couples by the set of cells that analyse them, a set named by its bits; the cells' counts and "
            f"the couples total are sums of these groups. {_GATE}",
            ordered=False,
            support=(*_GROUPS[:-1], "couples_by_cells_support"),
        )
    )
    for sex, one in _SEXES.items():
        histogram: Path = ("gates", f"mates_per_{one}")
        rules += [
            Table(histogram, f"The {sex} of the couples by their number of mates. {_GATE}", ordered=False),
            Derived(
                ("mating_pairs", f"n_{sex}_multiple_mates"),
                f"The {sex} with more than one mate, released only when every number of mates is held by no "
                f"{one} or by at least the minimum cell of them, so that the couples total beside it "
                "describes no small group of them, and only when the individuals in no couple are none or at "
                f"least the minimum cell: where it is zero the couples total is the number of {sex}.",
                sources=(histogram, _ROLES),
            ),
        ]
    rules += [
        Contingency(
            (*_GROUP_LEVELS, margin),
            "The smaller of the distinct mothers and distinct fathers of each group of couples at each level "
            f"of one mate's trait, the couples grouped by which of the cells with that margin analyse them. {_GATE}",
        )
        for margin in gates["couples_by_cells_levels"]
    ]
    links: list[Link] = list(
        shared_total("Every individual either has a value of the trait or lacks one.", _N, *trait_rows)
    )
    links += shared_total("Every individual is a mother of the couples, a father of them, or neither.", _N, _ROLES)
    links += [
        Link(
            (_ROLES, role),
            tuple(((*_ROWS_BY_ROLE, index), (role, state)) for state in _STATES),
            "The individuals of a role either have a value of the trait or lack one.",
        )
        for index in traits
        for role in gates["individuals_by_role"]
    ]
    links += [
        Link(
            ((*_TRAIT_ROWS, index), state),
            tuple(((*_ROWS_BY_ROLE, index), (role, state)) for role in gates["trait_rows_by_role"][index]),
            "An individual is a mother of the couples, a father of them, or neither.",
        )
        for index in traits
        for state in _STATES
    ]
    for key in (f"{mother}x{father}" for mother in traits for father in traits):
        cell = source["cells"][key]
        rules += _cell_rules(key, cell, gates["cells"][key])
        if "table" in cell:
            links += shared_total(
                "The couples of a cell's table are the couples the cell analyses.",
                ("cells", key, "n"),
                ("cells", key, "table"),
            )
    for sex in _SEXES:
        if sex in source.get("within_person", {}):
            rules += _within_person_rules(sex, gates["within_person"][sex])
    rules.append(Plain(("notes",), "Fixed notes on how to read the estimates, the same for every pedigree."))
    return ReleaseSpec(tuple(rules), tuple(links))


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------


def _record(name: str, released: Mapping[str, Any] | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One estimator's record, from its released fields and nothing else; ``extra`` precedes the permutation test."""
    key = VALUE_KEYS[name]
    extra = extra or {}
    if released is None:
        return {key: None, "reason": WITHHELD} | extra
    out: dict[str, Any] = {key: released.get(key)}
    if "reason" in released:
        return out | {"reason": released["reason"]} | extra
    out |= _draws(released, _ESTIMATE_FIELDS)
    if "bootstrap" in released:
        out["bootstrap"] = _draws(released["bootstrap"], _DRAWS_FIELDS)
    out |= extra
    out |= _draws(released, _TEST_FIELDS)
    if "permutations" in released:
        out["permutations"] = _draws(released["permutations"], _PERMUTATION_FIELDS)
    return out


def _cell(released: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        key: released[key] for key in ("mother", "father", "n", "n_mate_networks", "n_fixed_fathers") if key in released
    }
    crude: dict[str, Any] = {"table": released["table"]} if "table" in released else {}
    crude |= {name: _record(name, released["crude"][name]) for name in _ESTIMATORS if name in released.get("crude", {})}
    out["crude"] = crude
    if "stratified" in released:
        n_strata = released.get("n_strata") or {}
        counts = {f"n_strata_{sex}": n_strata.get(sex) for sex in _SEXES}
        out["stratified"] = {
            name: _record(name, released["stratified"][name], counts)
            for name in _ESTIMATORS
            if name in released["stratified"]
        }
    return out


def _within_person(released: Mapping[str, Any]) -> dict[str, Any]:
    name = released["estimator"]
    estimate = _record(name, released.get("estimate"))
    return {"estimator": name, "n": released.get("n")} | {
        key: estimate[key] for key in (VALUE_KEYS[name], "reason", "boundary") if key in estimate
    }


def safe_payload(source: Mapping[str, Any], min_cell: int) -> dict[str, Any]:
    """Release ``source`` at ``min_cell`` and lay the release out as ``assortative_mating.yaml``.

    Every value is read from the release at a rule's path: the meta block,
    which records ``min_cell``, then the ordinary file's sections with the
    leaves the list keeps. A withheld count is null; a withheld estimate is
    null with ``reason: withheld``. The gates are not laid out.
    """
    release = build_release(source, assortative_mating_spec(source), min_cell)
    rows = release.get("supports", {}).get("trait_rows", {})
    section: dict[str, Any] = {
        "traits": [
            {key: trait[key] for key in _TRAIT_FIELDS if key in trait} | {"rows": rows[index]}
            for index, trait in release["traits"].items()
        ],
        "settings": {key: release["settings"][key] for key in _SETTINGS if key in release.get("settings", {})},
        "inference": {key: release["inference"][key] for key in _INFERENCE if key in release.get("inference", {})},
        "mating_pairs": {
            key: release["mating_pairs"][key]
            for key in ("n_total", *(f"n_{sex}_multiple_mates" for sex in _SEXES))
            if key in release["mating_pairs"]
        },
        "mate_correlation": [_cell(cell) for cell in release["cells"].values()],
    }
    if "within_person" in release:
        section["within_person"] = {sex: _within_person(release["within_person"][sex]) for sex in _SEXES}
    section["notes"] = release["notes"]
    return {
        **{key: release[key] for key in (*_META, *_N) if key in release},
        "safe_attempt_min_cell": min_cell,
        _SECTION: section,
    }


__all__ = ["WITHHELD", "WORD_LEVELS", "assortative_mating_spec", "build_source", "safe_payload"]
