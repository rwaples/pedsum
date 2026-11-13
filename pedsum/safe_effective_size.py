"""``effective-size --safe-attempt``: the release list, and the payload it is applied to.

Safe mode writes nothing the ordinary ``effective_size.yaml`` payload holds.
:func:`build_source` builds a separate *source* payload: the counts each
estimator rests on, and the estimators' results grouped by the counts that
gate them. :func:`pedsum.disclosure.build_release` copies from it only the
leaves :func:`effective_size_spec` lists, and :func:`safe_payload` arranges
that release, and nothing else, into the file.

The counts are of two kinds. A *support* is a count the ordinary output
reports and a reader needs (individuals per depth and sex, the Hill
cohorts); it is released, protected at the minimum cell. A *gate* is a
count nothing reports (parents per depth, inbred individuals per depth); it
only decides whether an estimate is released, and is never written.

What the list leaves out, and why:

* Every extremum: the Hill cohort window's first and last birth year, its
  parental-age percentile and table, the census ratio.
* Every moment of a per-individual offspring count (``v_mm``, ``Vk_m``,
  ``kbar_m``, ...): the offspring-count tables they summarise are not
  released, so by the taint rule neither are they.
* Every number that repeats another: a mean of released counts, a count of
  the rows of a released table, a slope beside the Ne it gives.
* ``ne_long_term_contributions``, whatever the data: it is a concentration
  index over per-founder contributions, which narrows how the descendants
  are spread over the founders, and no count gate short of a dominance rule
  closes that. Its record is always ``reason: not_released_in_safe_mode``,
  and a safe run does not compute it.

A per-depth or per-cohort array is one leaf with its scalar, so both are
released or neither: an aggregate over a withheld array would give back the
withheld entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS

from pedsum.base import SEX_FEMALE, SEX_MALE
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
from pedsum.safe_summary import family_ids
from pedsum.sections import equivalent_complete_generations

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from pedigree_graph import PedigreeGraph

    from pedsum.disclosure import Path, Rule

_META = ("input", "command", "version", "generated_at")
_N: Path = ("n_total",)
_RUN = ("status", "estimators_requested")
_ESTIMATES = "effective_size"

_DEPTH: Path = ("supports", "depth")
_SEX: Path = ("supports", "sex")
_DEPTH_SEX: Path = ("supports", "depth_by_sex")
_REFERENCE_USE: Path = ("supports", "reference")
_COHORT_SEX: Path = ("supports", "birth_cohort_by_sex")
_PARENT_LINKS: Path = ("supports", "parent_links_with_birth_years")

_PARENTS: Path = ("gates", "parents_with_birth_years")
_GENOMES: Path = ("gates", "genomes_by_depth")
_COTWINS: Path = ("gates", "cotwins_by_depth")
_INBRED: Path = ("gates", "inbred_by_depth")
_RELATED: Path = ("gates", "related_families_by_depth")
_FATHERS: Path = ("gates", "fathers_by_depth")
_MOTHERS: Path = ("gates", "mothers_by_depth")
_USED: Path = ("gates", "delta_f_used_by_depth")
_REFERENCE: Path = ("gates", "reference_used")
_REFERENCE_INBRED: Path = ("gates", "reference_inbred")
_LAGGED: Path = ("gates", "reference_past_first_generation")
_COHORT_FATHERS: Path = ("gates", "fathers_by_birth_cohort")
_COHORT_MOTHERS: Path = ("gates", "mothers_by_birth_cohort")

#: The ``reason`` of an estimator whose result was withheld. A constant, not data.
WITHHELD = "withheld"

#: The ``reason`` of an estimator the release list has no field of. A constant, not data.
NOT_RELEASED = "not_released_in_safe_mode"

_SEX_DEPENDENT = ("ne_sex_ratio", "ne_variance_family_size", "ne_hill_overlapping")
_USES_F = ("ne_inbreeding", "ne_individual_delta_f", "ne_group_coancestry")
_USES_KINSHIP = ("ne_coancestry", "ne_group_coancestry")

# The keys of a record that did not run, copied as they are.
_UNAVAILABLE_KEYS = ("reason", "code", "rss_gib", "limit_gib")


# ---------------------------------------------------------------------------
# What each estimator releases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Field:
    """One field of an estimator's record; ``index`` names the record's label list an array is keyed by."""

    name: str
    index: str | None = None


@dataclass(frozen=True)
class _Group:
    """Fields of one estimator released together: all of them, or none, on the same counts."""

    key: str
    fields: tuple[_Field, ...]
    sources: tuple[Path, ...]
    reason: str


_NE = _Field("ne")
_CODE = _Field("code")
_TRANSITIONS = "transitions"
_FAMILY_SIZE: tuple[Path, ...] = (_DEPTH_SEX, _FATHERS, _MOTHERS)

_GROUPS: dict[str, tuple[_Group, ...]] = {
    "ne_inbreeding": (
        _Group(
            "estimate",
            (_NE, _CODE, _Field("mean_f_per_gen", "depths"), _Field("ne_per_gen", _TRANSITIONS)),
            (_DEPTH, _INBRED),
            "Mean F per depth and the Ne values computed from that series, released whole and only when every "
            "depth holds at least the minimum cell of individuals and its inbred are none or at least the "
            "minimum cell, so no mean is a sum over a few inbred individuals.",
        ),
    ),
    "ne_coancestry": (
        _Group(
            "estimate",
            (_NE, _CODE, _Field("mean_theta_per_gen", "depths"), _Field("ne_per_gen", _TRANSITIONS)),
            (_DEPTH, _GENOMES, _RELATED),
            "Mean kinship per depth and the Ne values computed from that series, released whole and only when "
            "every depth holds at least the minimum cell of genomes and its related pairs are none or come "
            "from at least the minimum cell of families and of shared parents, so no mean is the kinship "
            "among one parent's children.",
        ),
    ),
    "ne_variance_family_size": (
        _Group(
            "estimate",
            (_NE, _CODE, _Field("ne_per_transition", "parent_depths")),
            _FAMILY_SIZE,
            "Ne per parent depth and its harmonic mean, released whole and only when at every depth each sex "
            "reaches the minimum cell and its parents of each sex are none or at least the minimum cell; the "
            "offspring-count variances behind it are not released.",
        ),
    ),
    "ne_sex_ratio": (
        _Group(
            "estimate",
            (_NE, _CODE, _Field("ne_per_gen", "depths")),
            (_DEPTH_SEX,),
            "A function of the depth-by-sex table and nothing else, released only when that table is, so it "
            "adds no number to it.",
        ),
    ),
    "ne_individual_delta_f": (
        _Group(
            "estimate",
            (_NE, _CODE, _Field("standard_error")),
            (_REFERENCE, _REFERENCE_INBRED),
            "Ne from the mean individual increase in inbreeding over the reference individuals, with its "
            "standard error, released only when the reference individuals used and the inbred among them "
            "are each none or at least the minimum cell.",
        ),
        _Group(
            "unrelated_founders",
            (_Field("ne_unrelated_founders"),),
            (_REFERENCE, _REFERENCE_INBRED, _LAGGED),
            "The same Ne over the reference individuals more than one generation from the founders, released "
            "only when they too are none or at least the minimum cell.",
        ),
        _Group(
            "per_depth",
            (_Field("ne_per_gen", "depths"),),
            (_USED, _INBRED),
            "Ne per depth from the mean individual increase in inbreeding, released whole and only when at "
            "every depth the individuals used and the inbred among them are none or at least the minimum cell.",
        ),
    ),
    "ne_hill_overlapping": (
        _Group(
            "collapsed",
            (_NE, _CODE),
            _FAMILY_SIZE,
            "Without birth years this is the family-size variance Ne, released on that estimator's condition.",
        ),
        _Group(
            "interval",
            (_Field("T_m"), _Field("T_f")),
            (_PARENT_LINKS,),
            "Mean age of fathers and of mothers at their children's births, each a mean over the links of at "
            "least the minimum cell of distinct parents; no age, percentile or age table is released.",
        ),
        _Group(
            "estimate",
            (
                _NE,
                _CODE,
                _Field("Ne_m"),
                _Field("Ne_f"),
                _Field("ne_per_cohort", "cohort_years"),
                _Field("Ne_m_per_cohort", "cohort_years"),
                _Field("Ne_f_per_cohort", "cohort_years"),
            ),
            (_COHORT_SEX, _COHORT_FATHERS, _COHORT_MOTHERS, _PARENT_LINKS),
            "Ne per birth cohort and its harmonic means, released whole and only when every cohort used holds "
            "at least the minimum cell of each sex and of parents of each sex; the offspring-count moments "
            "and the cohort-size means are not released.",
        ),
    ),
    "ne_group_coancestry": (
        _Group(
            "estimate",
            (_NE, _CODE, _Field("mean_group_coancestry_per_gen", "depths"), _Field("ne_per_gen", _TRANSITIONS)),
            (_DEPTH, _GENOMES, _RELATED, _INBRED, _COTWINS),
            "Group coancestry per depth and the Ne values computed from that series, released whole and on "
            "the conditions of both the mean kinship and the mean F it is the sum of. Among unrelated "
            "individuals it is 1 / (2 x genomes), which beside the depth table counts the depth's MZ "
            "co-twins, so at every depth those are none or at least the minimum cell.",
        ),
    ),
}

#: The estimators a safe run computes: the ones with a released field.
RELEASABLE: tuple[str, ...] = tuple(name for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name in _GROUPS)

#: Fields of a record that say how the run was set up, copied as they are.
_OPTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "ne_individual_delta_f": (("reference_column", "The --reference-col option of the run, a column name."),),
    "ne_hill_overlapping": (
        ("collapses_to_ne_v", "Whether the run had no birth-year column, which is an option of the run."),
        ("vk_scaled", "An option of the estimator, the same for every pedigree."),
    ),
}


# ---------------------------------------------------------------------------
# Source payload
# ---------------------------------------------------------------------------


def _ran(record: Mapping[str, Any]) -> bool:
    """Whether the estimator ran: a record with no reason, or one that ran without an estimate."""
    return record.get("reason") in (None, "no_estimate")


def _labelled(labels: Sequence[Any], values: Sequence[Any]) -> dict[str, Any]:
    return {str(label): value for label, value in zip(labels, values, strict=True)}


def _index(record: Mapping[str, Any], name: str) -> list[str]:
    if name == _TRANSITIONS:
        return [f"{a}-{b}" for a, b in zip(record["transition_from"], record["transition_to"], strict=True)]
    return [str(label) for label in record[name]]


def _group_value(record: Mapping[str, Any], group: _Group) -> dict[str, Any] | None:
    """The fields of ``group`` that ``record`` holds, arrays keyed by their labels; None when it holds none."""
    out: dict[str, Any] = {}
    for field in group.fields:
        if field.name not in record:
            continue
        value = record[field.name]
        if field.index is not None and value is not None:
            value = _labelled(_index(record, field.index), value)
        out[field.name] = value
    return out or None


def _estimator_source(name: str, record: Mapping[str, Any]) -> dict[str, Any]:
    if not _ran(record):
        return {key: record[key] for key in _UNAVAILABLE_KEYS if record.get(key) is not None}
    out: dict[str, Any] = {key: record[key] for key, _ in _OPTIONS.get(name, ()) if key in record}
    groups = _GROUPS[name]
    if name == "ne_hill_overlapping":
        skip = ("interval", "estimate") if record["collapses_to_ne_v"] else ("collapsed",)
        groups = tuple(group for group in groups if group.key not in skip)
    for group in groups:
        value = _group_value(record, group)
        if value is not None:
            out[group.key] = value
    return out


def _by_depth(depth: np.ndarray, mask: np.ndarray | None, k: int) -> dict[str, int]:
    rows = depth if mask is None else depth[mask]
    return _labelled(range(k), np.bincount(rows, minlength=k).tolist())


def _same_depth_sibs(parent: np.ndarray, depth: np.ndarray, rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``rows`` sharing the recorded ``parent`` with another at their own depth, and the parents shared, by depth."""
    known = rows[parent[rows] >= 0]
    key = parent[known].astype(np.int64) * k + depth[known]
    shared, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return known[counts[inverse] > 1], np.bincount(shared[counts > 1] % k, minlength=k)


def _related_families(pg: PedigreeGraph, depth: np.ndarray, genome: np.ndarray, k: int) -> dict[str, int]:
    """A lower bound on the distinct units behind each depth's related pairs: families, and shared parents.

    Zero when the depth's mean kinship is zero: it then has no related
    pair. Otherwise the smaller of two counts over the individuals with a
    sib at their own depth, and at least one: their families, and the
    parents two of them share. Five half sibs by one father are five
    families and one shared parent; one couple's children are two shared
    parents and one family. Cousins at one depth add units this does not
    count, which errs toward withholding. Sorts the rows; builds no pair
    list.
    """
    summary = pg.mean_kinship_by_generation()
    related = np.zeros(k, dtype=bool)
    related[np.asarray(summary.generations)] = np.nan_to_num(np.asarray(summary.mean_kinship)) > 0
    mother, father = np.asarray(pg.mother_rows), np.asarray(pg.father_rows)
    rows = np.flatnonzero(genome)
    maternal, shared_mothers = _same_depth_sibs(mother, depth, rows, k)
    paternal, shared_fathers = _same_depth_sibs(father, depth, rows, k)
    sibs = np.union1d(maternal, paternal)
    family = family_ids(mother, father)
    families = np.bincount(np.unique(family[sibs].astype(np.int64) * k + depth[sibs]) % k, minlength=k)
    units = np.minimum(families, shared_mothers + shared_fathers)
    return _labelled(range(k), np.where(related, np.maximum(units, 1), 0).tolist())


def _counts(
    pg: PedigreeGraph,
    ran: Collection[str],
    results: Mapping[str, Mapping[str, Any]],
    reference_rows: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The supports and gates of the estimators in ``ran``, each from one pass over the rows."""
    depth = np.asarray(pg.depth)
    n = pg.n_individuals
    k = int(depth.max(initial=-1)) + 1
    mother, father = np.asarray(pg.mother_rows), np.asarray(pg.father_rows)
    twin = np.asarray(pg.twin_rows)
    # One row per genome, as pedigree-graph's genome-node cohorts keep the lower row of an MZ pair.
    genome = ~((twin >= 0) & (twin < np.arange(n)))

    supports: dict[str, Any] = {"depth": _by_depth(depth, None, k)}
    gates: dict[str, Any] = {"genomes_by_depth": _by_depth(depth, genome, k)}

    if any(name in ran for name in _SEX_DEPENDENT):
        sex = np.asarray(pg.sex)
        male, female = sex == SEX_MALE, sex == SEX_FEMALE
        is_father, is_mother = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        is_father[father[father >= 0]] = True
        is_mother[mother[mother >= 0]] = True
        males, females = _by_depth(depth, male, k), _by_depth(depth, female, k)
        n_male, n_female = int(male.sum()), int(female.sum())
        supports["sex"] = {"male": n_male, "female": n_female}
        supports["depth_by_sex"] = {d: {"male": males[d], "female": females[d]} for d in males}
        gates["fathers_by_depth"] = _by_depth(depth, male & is_father, k)
        gates["mothers_by_depth"] = _by_depth(depth, female & is_mother, k)

        hill = results["ne_hill_overlapping"]
        if "ne_hill_overlapping" in ran and not hill["collapses_to_ne_v"]:
            birth_year = np.asarray(pg.birth_year)
            known = birth_year >= 0
            links, parents = {}, {}
            for role, parent in (("father", father), ("mother", mother)):
                linked = parent[(parent >= 0) & known]
                linked = linked[known[linked]]
                links[role], parents[role] = int(linked.size), int(np.unique(linked).size)
            supports["parent_links_with_birth_years"] = links
            gates["parents_with_birth_years"] = parents
            years = hill.get("cohort_years")
            if years is not None:
                used_m, used_f = hill["N1_m_per_cohort"], hill["N1_f_per_cohort"]
                cohorts = {
                    str(year): {"male": n_m, "female": n_f}
                    for year, n_m, n_f in zip(years, used_m, used_f, strict=True)
                }
                # The rest of the pedigree is a row, so the table's total does not give its size by subtraction.
                cohorts["not_used"] = {"male": n_male - sum(used_m), "female": n_female - sum(used_f)}
                supports["birth_cohort_by_sex"] = cohorts
                for key, parents in (
                    ("fathers_by_birth_cohort", male & is_father),
                    ("mothers_by_birth_cohort", female & is_mother),
                ):
                    born, counts = np.unique(birth_year[parents & known], return_counts=True)
                    by_year = dict(zip(born.tolist(), counts.tolist(), strict=True))
                    gates[key] = {str(year): by_year.get(year, 0) for year in years}

    if any(name in ran for name in _USES_F):
        inbred = np.asarray(pg.inbreeding()) > 0
        gates["inbred_by_depth"] = _by_depth(depth, inbred, k)
        if "ne_individual_delta_f" in ran:
            delta_f = results["ne_individual_delta_f"]
            gates["delta_f_used_by_depth"] = _labelled(delta_f["depths"], delta_f["n_used_per_gen"])
            reference = np.flatnonzero(depth == k - 1) if reference_rows is None else reference_rows
            gates["reference_used"] = delta_f["n_reference"]
            # An inbred individual has related parents, so it is past the founders and is one the estimator uses.
            gates["reference_inbred"] = int(inbred[reference].sum())
            ecg = equivalent_complete_generations(pl.DataFrame({"ped_depth": depth}), mother, father)
            gates["reference_past_first_generation"] = int((ecg[reference] > 1.0).sum())
            if reference_rows is not None:
                supports["reference"] = {"used": delta_f["n_reference"], "not_used": n - delta_f["n_reference"]}

    if any(name in ran for name in _USES_KINSHIP):
        gates["related_families_by_depth"] = _related_families(pg, depth, genome, k)
    if "ne_group_coancestry" in ran:
        gates["cotwins_by_depth"] = _by_depth(depth, ~genome, k)
    return supports, gates


def build_source(
    *,
    meta: Mapping[str, Any],
    status: str,
    requested: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
    pg: PedigreeGraph | None,
    reference_rows: np.ndarray | None,
) -> dict[str, Any]:
    """The payload the release list is applied to: each estimator's results, and the counts they rest on.

    ``results`` are pedsum's records, one per estimator. From a record that
    ran, only the fields ``_GROUPS`` names are read, so a field
    pedigree-graph adds is not in the source. Without ``pg`` (a stop at the
    memory limit) there are no counts, and every estimate that needs one is
    withheld.
    """
    ran = [name for name in requested if _ran(results[name])]
    supports, gates = _counts(pg, ran, results, reference_rows) if pg is not None and ran else ({}, {})
    return {
        **{key: meta[key] for key in (*_META, *_N)},
        "status": status,
        "estimators_requested": list(requested),
        "supports": supports,
        "gates": gates,
        _ESTIMATES: {name: _estimator_source(name, results[name]) for name in RELEASABLE},
    }


# ---------------------------------------------------------------------------
# Release list
# ---------------------------------------------------------------------------

_GATE = "It decides whether estimates are released and is not written to the file."


def _rules() -> tuple[Rule, ...]:
    rules: list[Rule] = [
        Plain((key,), "Describes the run, not the individuals: the command, its input path, version and time.")
        for key in _META
    ]
    rules += [
        Count(_N, "The number of individuals in the pedigree, withheld when below the minimum cell."),
        Plain(("status",), "Whether the run finished or stopped at its memory limit."),
        Plain(("estimators_requested",), "The --estimators option of the run, which the data does not set."),
        Table(
            _DEPTH,
            "Individuals by depth, the cohorts every per-depth estimate is over; a small deepest or shallowest "
            "depth is merged into its neighbour, and every per-depth estimate is then withheld.",
            ordered=True,
        ),
        Table(
            _SEX,
            "Individuals by sex, both at the minimum cell or the table is withheld; the margin the two "
            "by-sex tables share.",
            ordered=False,
        ),
        Contingency(
            _DEPTH_SEX,
            "Individuals by depth and sex, the counts the sex-ratio and family-size estimators use; released "
            "whole, and only when no cell is small.",
        ),
        Table(
            _REFERENCE_USE,
            "Individuals by whether the estimator averages over them under --reference-col; both classes at "
            "the minimum cell, or the table is withheld.",
            ordered=False,
        ),
        Contingency(
            _COHORT_SEX,
            "Individuals by sex and the birth cohort the Hill estimator uses them in, the rest of the pedigree "
            "in one row; released whole, and only when no cell is small, so each year shown is shared by at "
            "least the minimum cell of each sex and is no individual's extreme.",
        ),
        Table(
            _PARENT_LINKS,
            "Father links and mother links whose two birth years are both known, the links the mean parental "
            "ages are over; the links of each sex must come from at least the minimum cell of distinct "
            "parents, or the table is withheld.",
            ordered=False,
            support=_PARENTS,
        ),
    ]
    rules += [
        Table(path, f"{what}. {_GATE}", ordered=False)
        for path, what in (
            (_GENOMES, "Genomes by depth, counting MZ co-twins once"),
            (_COTWINS, "MZ co-twins a depth's genomes leave out, by depth"),
            (_INBRED, "Inbred individuals by depth"),
            (_RELATED, "A lower bound on the families, and on the shared parents, behind each depth's related pairs"),
            (_FATHERS, "Males with recorded offspring, by depth"),
            (_MOTHERS, "Females with recorded offspring, by depth"),
            (_USED, "Individuals past the founders, by depth"),
            (_COHORT_FATHERS, "Males with recorded offspring, by birth cohort used"),
            (_COHORT_MOTHERS, "Females with recorded offspring, by birth cohort used"),
        )
    ]
    rules += [
        Count(path, f"{what}. {_GATE}")
        for path, what in (
            (_REFERENCE, "Reference individuals past the founders, the ones the estimator averages over"),
            (_REFERENCE_INBRED, "Reference individuals whose F exceeds zero"),
            (_LAGGED, "Reference individuals more than one generation from the founders"),
        )
    ]
    for name in RELEASABLE:
        rules += [
            Plain(
                (_ESTIMATES, name, key),
                "Why the estimator did not run: it was not selected, the input lacks a column it needs, or "
                "the run stopped at its memory limit.",
            )
            for key in _UNAVAILABLE_KEYS
        ]
        rules += [Plain((_ESTIMATES, name, key), reason) for key, reason in _OPTIONS.get(name, ())]
        rules += [
            Derived((_ESTIMATES, name, group.key), group.reason, sources=group.sources) for group in _GROUPS[name]
        ]
    return tuple(rules)


_RULES = _rules()


def effective_size_spec(source: Mapping[str, Any]) -> ReleaseSpec:
    """The release list for ``effective-size``, with the links among the released tables ``source`` holds.

    The rules are fixed. The links name table cells, and the depths are
    read from ``source``. Only released tables are linked: a gate is not
    written, so a reader has no cell of it to combine.
    """
    supports = source["supports"]
    splits = [path for path in (_DEPTH, _SEX, _COHORT_SEX, _REFERENCE_USE) if path[-1] in supports]
    links = list(shared_total("Every individual is counted once in each of these tables.", _N, *splits))
    links += [
        Link(
            (_DEPTH, depth),
            tuple((_DEPTH_SEX, (depth, sex)) for sex in cells),
            "A depth's males and females are its individuals: the estimators that use sex need it for every row.",
        )
        for depth, cells in supports.get("depth_by_sex", {}).items()
    ]
    for table in (_DEPTH_SEX, _COHORT_SEX):
        rows = supports.get(table[-1], {})
        links += [
            Link((_SEX, sex), tuple((table, (row, sex)) for row in rows), "A column of the table is one sex.")
            for sex in supports.get("sex", {})
            if rows
        ]
    return ReleaseSpec(_RULES, tuple(links))


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------


def _record(name: str, released: Mapping[str, Any]) -> dict[str, Any]:
    """One estimator's record, from the released leaves and nothing else."""
    if "reason" in released:
        return {"ne": None} | {key: released[key] for key in _UNAVAILABLE_KEYS if key in released}
    out: dict[str, Any] = {"ne": None}
    for group in _GROUPS[name]:
        if group.key not in released:
            continue
        value = released[group.key]
        has_ne = group.fields[0] is _NE
        if value is None:
            if has_ne:
                out["reason"] = WITHHELD
            out |= {field.name: None for field in group.fields if field is not _CODE}
        else:
            if has_ne and value.get("ne") is None:
                out["reason"] = "no_estimate"
            out |= {field.name: value[field.name] for field in group.fields if field.name in value}
    return out | {key: released[key] for key, _ in _OPTIONS.get(name, ()) if key in released}


def safe_payload(source: Mapping[str, Any], min_cell: int) -> dict[str, Any]:
    """Release ``source`` at ``min_cell`` and lay the release out as ``effective_size.yaml``.

    Every value is read from the release at a rule's path: the meta block,
    which records ``min_cell``, the supports, then one record per
    estimator. A withheld estimate is null with ``reason: withheld``, and
    an estimator outside ``RELEASABLE`` has the same record whatever the
    run. The gates are not laid out.
    """
    release = build_release(source, effective_size_spec(source), min_cell)
    estimates = release.get(_ESTIMATES, {})
    return {
        **{key: release[key] for key in (*_META, *_N) if key in release},
        "safe_attempt_min_cell": min_cell,
        **{key: release[key] for key in _RUN if key in release},
        "supports": {
            path[-1]: release["supports"][path[-1]]
            for path in (_DEPTH, _SEX, _DEPTH_SEX, _REFERENCE_USE, _COHORT_SEX, _PARENT_LINKS)
            if path[-1] in release.get("supports", {})
        },
        _ESTIMATES: {
            name: _record(name, estimates.get(name, {})) if name in _GROUPS else {"ne": None, "reason": NOT_RELEASED}
            for name in ALL_EFFECTIVE_SIZE_ESTIMATORS
        },
    }


__all__ = ["NOT_RELEASED", "RELEASABLE", "WITHHELD", "build_source", "effective_size_spec", "safe_payload"]
