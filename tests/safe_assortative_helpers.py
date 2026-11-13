"""Pedigrees, runs and the reader's system for the ``assortative-mating --safe-attempt`` tests.

:func:`reader_system` is what a reader of the safe file can write down.
Beyond the released tables and the declared links it models the couples
themselves: one unknown per *pattern* of the two mates' trait states (a
level, a value, or missing) and the set of cells that analyse the couple.
Every released count of couples is a sum of those unknowns, so two tables
over the same couples share their margins without a link per margin. Each
sex's mates-per-person histogram is modelled beside it: the couples are its
first moment, the distinct persons its total. The individuals with and
without each trait are split into mothers, fathers and everyone else, tied
to the couples whose mate lacks the trait, and a cell's Mate Networks are
bounded by its couples and the extra mates.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml
from conftest import run_pedsum, written_output
from disclosure_oracle import add_nonlinear, system_from_release
from test_safe_summary import at, leaves

import pedsum.assortative_mating as ordinary
from pedsum import safe_assortative_mating
from pedsum.disclosure import Contingency, Count, Table, build_release, protect_payload
from pedsum.safe_assortative_mating import assortative_mating_spec, build_source

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable
    from pathlib import Path

    from disclosure_oracle import System

    from pedsum.disclosure import ReleaseSpec

AM = "assortative_mating"
PAIRS = ("mating_pairs", "n_total")
GROUPS = ("gates", "couples_by_cells")
GROUP_LEVELS = ("gates", "couples_by_cells_levels")
BY_ROLE = ("gates", "trait_rows_by_role")

#: Integer leaves that are not counts of units: run options, replicate counts, and counts of strata.
NOT_COUNTS = {
    *("safe_attempt_min_cell", "permutations", "bootstrap", "seed", "threads", "birth_year_bin"),
    *("min_stratum_networks", "permutation_stop_h", "requested", "valid", "failed", "draws_used", "sequential_h"),
    *("n_strata_mothers", "n_strata_fathers"),
}


def half_width(value: float) -> float:
    """Half the last digit ``assortative_mating.yaml`` keeps of ``value``: 4 decimals, or 4 figures when that is more."""
    if not value or not math.isfinite(value):
        return 0.0
    return 0.5 * 10.0 ** -max(4, 3 - math.floor(math.log10(abs(value))))


def couple_rows(couples: list[tuple[int, int]], children: bool = False, **traits: dict[int, Any]) -> list[dict]:
    """A two-generation pedigree with one child per couple; ``traits`` maps a column to each parent's value.

    A parent without a value is ``NA``, and so is every child unless
    ``children``, which gives them alternating levels 0 and 1. Birth years
    put the parents in 1950 and the children in 1980.
    """
    people = sorted({mother for mother, _ in couples}) + sorted({father for _, father in couples})
    mothers = {mother for mother, _ in couples}
    rows = [
        {"id": person, "sex": "F" if person in mothers else "M", "mother": -1, "father": -1, "birth_year": 1950}
        | {name: str(values.get(person, "NA")) for name, values in traits.items()}
        for person in people
    ]
    rows += [
        {"id": 100_000 + i, "sex": "MF"[i % 2], "mother": mother, "father": father, "birth_year": 1980}
        | dict.fromkeys(traits, str(i % 2) if children else "NA")
        for i, (mother, father) in enumerate(couples)
    ]
    return rows


def table_couples(table: list[list[int]], start: int = 1) -> tuple[list[tuple[int, int]], dict[int, int]]:
    """Couples of distinct people whose levels give ``table[mother level][father level]``, and each person's level."""
    couples, level = [], {}
    person = start
    for m_level, row in enumerate(table):
        for f_level, count in enumerate(row):
            for _ in range(count):
                mother, father = person, 50_000 + person
                couples.append((mother, father))
                level[mother], level[father] = m_level, f_level
                person += 1
    return couples, level


def mating_rows(
    seed: int = 11,
    founders: int = 80,
    per_generation: int = 160,
    generations: int = 3,
    missing: float = 0.0,
    remating: int = 6,
) -> list[dict]:
    """Generations of couples with three traits: ``dx`` (no/yes), ``grade`` (0, 1, 2) and ``score``.

    Each generation's women and men pair off at random, and ``remating`` of
    the women have a child with a second man, who has his own mate too.
    Each trait is ``NA`` with probability ``missing``.
    """
    rng = np.random.default_rng(seed)

    def person(i: int, mother: int, father: int, generation: int) -> dict:
        liability = rng.normal()
        values = {
            "dx": "yes" if liability > 0.4 else "no",
            "grade": str(int(np.digitize(liability + rng.normal(), [-0.5, 0.8]))),
            "score": f"{liability + rng.normal():.3f}",
        }
        return {
            "id": i,
            "sex": "MF"[i % 2],
            "mother": mother,
            "father": father,
            "birth_year": 1900 + 30 * generation + i % 7,
        } | {name: "NA" if rng.random() < missing else value for name, value in values.items()}

    rows = [person(i + 1, -1, -1, 0) for i in range(founders)]
    previous = rows
    for generation in range(1, generations + 1):
        dams = rng.permutation([row["id"] for row in previous if row["sex"] == "F"]).tolist()
        sires = rng.permutation([row["id"] for row in previous if row["sex"] == "M"]).tolist()
        couples = list(zip(dams, sires, strict=False))
        couples += [(dams[2 * i], sires[2 * i + 1]) for i in range(min(remating, len(couples) // 2))]
        parents = couples + [couples[int(rng.integers(len(couples)))] for _ in range(per_generation - len(couples))]
        current = [person(len(rows) + i + 1, mother, father, generation) for i, (mother, father) in enumerate(parents)]
        rows += current
        previous = current
    return rows


def run(
    ped: Path, out_dir: Path, *flags: str, safe: bool = True, edit=None, records=None, code: int | None = 0
) -> dict[str, Any] | None:
    """Run ``assortative-mating`` and return the safe source payload it built (None for an ordinary run).

    ``edit`` changes the source before it is released; ``records`` changes
    the ordinary payload before the source is built from it. ``code`` is
    the exit code the run must have, or None for a run that may fail.
    """
    captured: list[dict[str, Any]] = []

    def capture(**kwargs):
        source = build_source(**kwargs)
        if edit is not None:
            edit(source)
        captured.append(source)
        return source

    def compute(*args, **kwargs):
        payload = real_compute(*args, **kwargs)
        if records is not None:
            records(payload)
        return payload

    real_compute = ordinary.compute_assortative_mating
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(safe_assortative_mating, "build_source", capture)
        mp.setattr(ordinary, "compute_assortative_mating", compute)
        args = ["assortative-mating", "--in", str(ped), "--out", str(out_dir), "--threads", "1", *flags]
        res = run_pedsum([*args, *(["--safe-attempt"] if safe else [])])
    assert code is None or res.returncode == code, res.stderr
    return captured[0] if safe and captured else None


def load(out_dir: Path) -> dict[str, Any]:
    """The ``assortative_mating.yaml`` of a run."""
    return yaml.safe_load(written_output(out_dir, "assortative_mating.yaml").read_text())


def body(payload: dict[str, Any]) -> str:
    """The section of a payload as text, without the meta block's paths and times."""
    return yaml.safe_dump(payload[AM])


def small_counts(payload: dict[str, Any], min_cell: int) -> list[tuple[str, ...]]:
    """Paths of every integer of the file that is in ``1..min_cell-1`` and is not a known non-count."""
    return [
        path
        for path, leaf in leaves(payload)
        if isinstance(leaf, int) and not isinstance(leaf, bool) and 0 < leaf < min_cell and path[-1] not in NOT_COUNTS
    ]


def cell_record(payload: dict[str, Any], mother: str, father: str) -> dict[str, Any]:
    """The ``mate_correlation`` record of one cell of a written file."""
    (record,) = [r for r in payload[AM]["mate_correlation"] if (r["mother"], r["father"]) == (mother, father)]
    return record


# ---------------------------------------------------------------------------
# The reader's system
# ---------------------------------------------------------------------------


def protected_cell(cell, source: dict[str, Any], spec: ReleaseSpec, min_cell: int) -> bool:
    """Whether ``cell`` is one the minimum-cell rule protects, by its count and the people behind it."""
    path, label = cell
    rule = next(rule for rule in spec.rules if rule.path == path)
    value = at(source, path)
    if isinstance(rule, Count):
        return True
    support = None if rule.support is None else at(source, rule.support)
    if isinstance(rule, Table):
        gate = value[label] if support is None else min(value[label], support[label])
        return 0 < gate < min_cell
    assert isinstance(rule, Contingency)
    row, col = label
    if value[row][col]:
        gate = value[row][col] if support is None else min(value[row][col], support[row][col])
        return gate < min_cell
    totals = (sum(value[row].values()), sum(cells[col] for cells in value.values()))
    return 0 not in totals and any(total < min_cell for total in totals)


def _states(source: dict[str, Any]) -> list[list[Any]]:
    """Per trait, the states a mate can be in: each level of a discrete trait or ``v`` for a value, and ``.``."""
    out = []
    for index in source["traits"]:
        table = source["cells"][f"{index}x{index}"].get("table")
        out.append([*(table if table is not None else ["v"]), "."])
    return out


def _closed_forms(release: dict[str, Any]) -> list:
    """Checks a 2x2 ``(a, b, c, d)`` must pass to serialise to each released closed-form estimate of its table."""
    checks = []
    odds_ratio = (release.get("odds_ratio") or {}).get("value")
    if odds_ratio is not None:

        def gives_odds_ratio(a: int, b: int, c: int, d: int) -> bool:
            if (a + b) * (c + d) * (a + c) * (b + d) == 0:
                return False
            if b * c == 0:
                return odds_ratio == math.inf
            return abs(a * d / (b * c) - odds_ratio) <= half_width(odds_ratio) + 1e-12

        checks.append(gives_odds_ratio)
    phi = (release.get("phi") or {}).get("r")
    if phi is not None:

        def gives_phi(a: int, b: int, c: int, d: int) -> bool:
            margins = (a + b) * (c + d) * (a + c) * (b + d)
            return margins > 0 and abs((a * d - b * c) / math.sqrt(margins) - phi) <= half_width(phi) + 1e-12

        checks.append(gives_phi)
    return checks


def reader_system(
    source: dict[str, Any], min_cell: int, spec: ReleaseSpec | None = None
) -> tuple[System, list[Hashable]]:
    """What a reader of the safe file can write down, and the protected numbers withheld from them.

    The targets are every protected cell of a withheld table, every withheld
    count, every small group of couples (by the cells that analyse them),
    every small level of such a group, every small class of a
    mates-per-person histogram and every small part of a trait's rows among
    the mothers, the fathers and everyone else.
    """
    spec = spec or assortative_mating_spec(source)
    rules = {rule.path: rule for rule in spec.rules}
    release = build_release(source, spec, min_cell)
    protected = protect_payload(source, spec, min_cell)
    # Every couple has a child, so no count of couples or of parents exceeds the individuals.
    cap = max(source["n_total"], 3)

    # A count withheld for the people behind it is not itself below the minimum cell.
    free = [
        path
        for path, table in protected.items()
        if path[0] != "gates" and isinstance(rules[path], Count) and rules[path].support and table.render() is None
    ]
    visible = {path: table for path, table in protected.items() if path[0] != "gates" and path not in free}
    system, hidden = system_from_release(spec, visible, min_cell, cap=cap)
    targets: list[Hashable] = [cell for cell in hidden if protected_cell(cell, source, spec, min_cell)]
    for path in free:
        system.var((path, None), 0, cap)
        targets.append((path, None))

    def table_cells(path: tuple[str, ...]) -> list:
        return [(path, (r, c)) for r, cols in at(source, path).items() for c in cols]

    for link in spec.links:
        (whole, _), (part, _) = link.whole, link.parts[0]
        if whole in free and part in visible:
            system.add({(whole, None): 1} | dict.fromkeys(table_cells(part), -1), 0, 0)

    states = _states(source)
    cells = list(source["cells"])
    stratified = at(source, ("settings", "stratify_by")) is not None
    joint: dict[tuple, list[Hashable]] = {}
    for pattern in itertools.product(*states, *states):
        mother, father = pattern[: len(states)], pattern[len(states) :]
        complete = [key for key in cells if mother[int(key[0])] != "." and father[int(key[-1])] != "."]
        subsets = (
            itertools.chain.from_iterable(itertools.combinations(complete, k) for k in range(len(complete) + 1))
            if stratified
            else [tuple(complete)]
        )
        for analysed in subsets:
            name = ("couples", pattern, analysed)
            system.var(name, 0, cap)
            joint[pattern, analysed] = name

    def couples(keep) -> dict[Hashable, int]:
        return {name: 1 for (pattern, analysed), name in joint.items() if keep(pattern, analysed)}

    def side(pattern: tuple, key: str) -> tuple[Any, Any]:
        return pattern[int(key[0])], pattern[len(states) + int(key[-1])]

    everyone = couples(lambda *_: True)
    unknown_stratum = ("couples", "unknown stratum")
    system.var(unknown_stratum, 0, cap if stratified else 0)
    system.add({(PAIRS, None): 1, unknown_stratum: -1} | dict.fromkeys(everyone, -1), 0, 0)

    for key in cells:
        n = (("cells", key, "n"), None)
        system.add({n: 1} | dict.fromkeys(couples(lambda _, analysed, key=key: key in analysed), -1), 0, 0)
        table = ("cells", key, "table")
        if table not in protected:
            continue
        for cell in table_cells(table):
            members = couples(lambda p, analysed, key=key, cell=cell: key in analysed and side(p, key) == cell[1])
            system.add({cell: 1} | dict.fromkeys(members, -1), 0, 0)
        shape = (len(at(source, table)), len(at(source, table)["0"]))
        total = protected[table].total
        checks = _closed_forms(at(release, ("cells", key, "crude")) or {})
        if protected[table].counts is None and checks and shape == (2, 2) and total is not None:
            candidates = [
                (a, b, c, total - a - b - c)
                for a in range(total + 1)
                for b in range(total - a + 1)
                for c in range(total - a - b + 1)
            ]
            add_nonlinear(
                system, table_cells(table), candidates, lambda *t, checks=checks: all(check(*t) for check in checks)
            )

    for index in source["traits"]:
        missing = ((*("supports", "trait_rows"), index), "missing")
        for offset in (0, len(states)):
            lacking = couples(lambda p, _, at_=offset + int(index): p[at_] == ".")
            system.add({missing: -cap} | lacking, -cap * cap, 0)

    groups = at(source, GROUPS)
    support = at(source, (*GROUPS[:-1], "couples_by_cells_support"))
    for mask, count in groups.items():
        if not 0 < min(count, support[mask]) < min_cell:
            continue
        analysed = tuple(key for bit, key in enumerate(cells) if int(mask) >> bit & 1)
        members = couples(lambda _, a, analysed=analysed: a == analysed)
        name = ("group", mask)
        system.var(name, 0, cap)
        system.add({name: 1} | dict.fromkeys(members, -1) | ({} if analysed else {unknown_stratum: -1}), 0, 0)
        targets.append(name)

    for link in spec.links:
        if link.parts[0][0][:2] != BY_ROLE:
            continue
        for part in link.parts:
            system.var(part, 0, cap)
            if 0 < at(source, part[0])[part[1][0]][part[1][1]] < min_cell:
                targets.append(part)
        system.add({link.whole: 1} | dict.fromkeys(link.parts, -1), 0, 0)

    extra_mates: dict[Hashable, int] = {}
    for offset, (sex, one) in zip((0, len(states)), (("mothers", "mother"), ("fathers", "father")), strict=True):
        histogram = at(source, ("gates", f"mates_per_{one}"))
        classes = range(1, max(map(int, histogram), default=1) + 3)
        for k in classes:
            system.var((sex, k), 0, cap)
            if 0 < histogram.get(str(k), 0) < min_cell:
                targets.append((sex, k))
        persons = {(sex, k): 1 for k in classes}
        extra = {(sex, k): k - 1 for k in classes}
        extra_mates |= extra
        system.add({(sex, k): k for k in classes} | dict.fromkeys(everyone, -1), 0, 0)
        several = at(release, ("mating_pairs", f"n_{sex}_multiple_mates"))
        if several is not None:
            system.add({(sex, k): 1 for k in classes if k > 1}, several, several)

        # Each person lacking a trait is in at least one couple, and in one more per extra mate.
        without = {index: ((*BY_ROLE, index), (sex, "missing")) for index in source["traits"]}
        for index, missing in without.items():
            system.add(persons | {((*BY_ROLE, index), (sex, state)): -1 for state in ("with_value", "missing")}, 0, 0)
            lacking = couples(lambda p, _, at_=offset + int(index): p[at_] == ".")
            system.add(lacking | {missing: -1}, 0, cap)
            system.add(lacking | {missing: -1} | {name: -weight for name, weight in extra.items()}, -cap, 0)
        if at(source, ("within_person", sex)) is not None:
            lacking_any = (sex, "lacking a trait")
            system.var(lacking_any, 0, cap)
            n = (("within_person", sex, "n"), None)
            system.add(persons | {n: -1, lacking_any: -1}, 0, 0)
            system.add({lacking_any: 1} | dict.fromkeys(without.values(), -1), -cap, 0)
            for missing in without.values():
                system.add({lacking_any: 1, missing: -1}, 0, cap)

    # A network holds at least one couple, and each couple beyond the first is an extra mate of someone.
    for key in cells:
        networks = ("cells", key, "n_mate_networks")
        if networks in protected:
            beyond = {(("cells", key, "n"), None): 1, (networks, None): -1}
            system.add(beyond, 0, cap)
            system.add(beyond | {name: -weight for name, weight in extra_mates.items()}, -2 * cap, 0)

    for margin, table in at(source, GROUP_LEVELS).items():
        one, index = margin.split("_")
        position = int(index) + (len(states) if one == "father" else 0)
        sharing = [key for key in cells if key[0 if one == "mother" else -1] == index]
        for mask, row in table.items():
            analysed = {key for bit, key in enumerate(cells) if int(mask) >> bit & 1}
            for level, support in row.items():
                if 0 < support < min_cell:
                    name = ("group level", mask, margin, level)
                    members = couples(
                        lambda p, a, analysed=analysed, level=level, at_=position, sharing=sharing: (
                            analysed == set(a).intersection(sharing) and p[at_] == level
                        )
                    )
                    system.var(name, 0, cap)
                    system.add({name: 1} | dict.fromkeys(members, -1), 0, 0)
                    targets.append(name)
    return system, targets


def pinned_protected(source: dict[str, Any], min_cell: int, spec: ReleaseSpec | None = None) -> list[Hashable]:
    """The protected numbers a reader of the safe file can compute exactly."""
    system, targets = reader_system(source, min_cell, spec)
    return system.pinned(targets)


def every_cell(source: dict[str, Any], paths: Iterable[tuple[str, ...]]) -> list:
    """Each ``(path, label)`` cell of the contingency tables at ``paths``."""
    return [(path, (r, c)) for path in paths for r, cols in at(source, path).items() for c in cols]
