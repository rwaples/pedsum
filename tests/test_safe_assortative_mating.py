"""``assortative-mating --safe-attempt``: the release list, audit leak 7, the couple supports, and the exact oracle.

``mating_rows`` is a random-mating pedigree with a binary, an ordinal and a
continuous trait, large enough that a safe run releases every estimate.
``couple_rows`` builds a pedigree from a list of couples, for the cases
that need an exact table or an exact set of rematings.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from conftest import property_budget, run_pedsum, write_ped
from disclosure_oracle import System, add_nonlinear
from hypothesis import given
from hypothesis import strategies as st
from safe_assortative_helpers import (
    AM,
    BY_ROLE,
    GROUP_LEVELS,
    GROUPS,
    ROLES,
    body,
    cell_record,
    couple_rows,
    every_cell,
    funnel_rows,
    half_width,
    load,
    mating_rows,
    pinned_protected,
    reader_system,
    run,
    small_counts,
    table_couples,
)
from test_safe_summary import _cell_sum, at, leaves

import pedsum.assortative_mating as ordinary_module
from pedsum.assortative_mating import VALUE_KEYS
from pedsum.disclosure import DEFAULT_MIN_CELL, Count, Derived, ReleaseSpec
from pedsum.safe_assortative_mating import (
    WITHHELD,
    WORD_LEVELS,
    _Couples,
    assortative_mating_spec,
    safe_payload,
)

TWO = ("--trait", "dx", "score", "--permutations", "39", "--seed", "3")
ORDINAL = ("--trait", "grade", "dx", "--trait-type", "grade=ordinal", "--permutations", "39")
STRATA = ("--stratify-by", "depth", "--min-stratum-networks", "5")


@pytest.fixture(scope="module")
def wide_ped(tmp_path_factory) -> Path:
    """The random-mating pedigree, written once."""
    return write_ped(tmp_path_factory.mktemp("wide") / "wide.tsv", mating_rows())


@pytest.fixture(scope="module")
def wide(wide_ped, tmp_path_factory) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The safe source, the safe file and the ordinary file of the two-trait stratified run."""
    out_dir = tmp_path_factory.mktemp("wide_out")
    source = run(wide_ped, out_dir / "safe", *TWO, *STRATA)
    assert source is not None
    assert [path.name for path in (out_dir / "safe").iterdir()] == ["assortative_mating.txt"]
    run(wide_ped, out_dir / "plain", *TWO, *STRATA, safe=False)
    return source, load(out_dir / "safe"), load(out_dir / "plain")


@pytest.fixture(scope="module")
def binary(wide_ped, tmp_path_factory) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The safe source, the safe file and the ordinary file of the one-trait binary run."""
    out_dir = tmp_path_factory.mktemp("binary_out")
    flags = ("--trait", "dx", "--permutations", "39", "--bootstrap", "50")
    source = run(wide_ped, out_dir / "safe", *flags)
    assert source is not None
    run(wide_ped, out_dir / "plain", *flags, safe=False)
    return source, load(out_dir / "safe"), load(out_dir / "plain")


# ---------------------------------------------------------------------------
# Release list
# ---------------------------------------------------------------------------


def test_every_rule_has_a_reason(wide, binary):
    """Each released leaf carries a sentence, not a placeholder."""
    for source in (wide[0], binary[0]):
        for rule in assortative_mating_spec(source).rules:
            assert len(rule.reason.split()) >= 6, rule.path
            assert rule.reason.endswith("."), rule.path


def test_the_file_holds_the_release_and_no_gate(wide):
    """The file's keys are fixed, every cell has a record, and no gate is written."""
    source, written, ordinary = wide
    assert list(written) == ["input", "command", "version", "generated_at", "n_total", "safe_attempt_min_cell", AM]
    assert list(written[AM]) == [
        *("traits", "settings", "inference", "mating_pairs", "mate_correlation", "within_person", "notes"),
    ]
    assert [(r["mother"], r["father"]) for r in written[AM]["mate_correlation"]] == [
        (r["mother"], r["father"]) for r in ordinary[AM]["mate_correlation"]
    ]
    text = body(written)
    for name in ("gates", "floor", "support", "couples_by_cells", "mates_per", "_strata:", "stratum_levels"):
        assert name not in text, name
    assert source["gates"]["cells"]["0x0"]["mother_stratum_levels"]


def test_a_key_no_rule_names_is_absent_from_the_file(wide_ped, tmp_path):
    """A field pg-phenotype adds to a record, and a key added anywhere in the source, reach the file nowhere."""
    marker = "INJECTED-7c1f"

    def new_payload_fields(payload: dict[str, Any]) -> None:
        payload["new_section"] = {"n": 1, "note": marker}
        payload["mating_pairs"]["new_count"] = 1
        payload["settings"]["new_setting"] = marker
        payload["inference"]["new_text"] = marker
        for cell in payload["mate_correlation"]:
            cell["new_field"] = marker
            cell["crude"]["new_estimator"] = {"r": 0.5, "note": marker}
            for record in (*cell["crude"].values(), *cell.get("stratified", {}).values()):
                if isinstance(record, dict):
                    record["new_field"] = marker
                    record.get("permutations", {})["new_draws"] = marker
                    record.get("bootstrap", {})["new_draws"] = marker
        for record in payload["within_person"].values():
            record["new_field"] = marker

    def inject(source: dict[str, Any]) -> None:
        rules = {rule.path: rule for rule in assortative_mating_spec(source).rules}

        def walk(node: dict[str, Any], path: tuple[str, ...]) -> None:
            # Into each estimator's record, but not into a table: its keys are its labels.
            for key, value in list(node.items()):
                rule = rules.get((*path, key))
                if isinstance(value, dict) and (rule is None or isinstance(rule, Derived)):
                    walk(value, (*path, key))
            node[f"new_key_{len(path)}"] = marker
            node[f"new_section_{len(path)}"] = {"n": 1, "note": marker}

        for key in ("traits", "settings", "inference", "mating_pairs", "cells", "within_person", "supports"):
            walk(source[key], (key,))
        source["new_top"] = marker

    source = run(wide_ped, tmp_path, *TWO, *STRATA, "--bootstrap", "20", records=new_payload_fields, edit=inject)
    assert source is not None
    assert "new_field" not in str({key: value for key, value in source.items() if key != "new_top"}).replace(
        "new_key", ""
    ).replace("new_section", "")
    assert marker in str(source["cells"]["0x0"]["crude"]["tetrachoric"])
    assert marker in str(source["within_person"]["mothers"]["estimate"])
    written = (tmp_path / "assortative_mating.txt").read_text()
    assert cell_record(load(tmp_path), "dx", "dx")["crude"]["tetrachoric"]["rho"] is not None
    assert marker not in written
    assert "new_" not in written


def test_links_are_true_and_cover_every_shared_total(wide):
    """Each link holds in the data: a trait's rows are the individuals and split by role, a cell's table is its couples."""
    source, _, _ = wide
    spec = assortative_mating_spec(source)
    for link in spec.links:
        assert link.reason.endswith("."), link
        assert _cell_sum(source, link.whole) == sum(_cell_sum(source, part) for part in link.parts), link
    rows = ("supports", "trait_rows")
    assert {(link.whole[0], link.parts[0][0]) for link in spec.links} == {
        (("n_total",), (*rows, "0")),
        (("n_total",), (*rows, "1")),
        (("n_total",), ROLES),
        (ROLES, (*BY_ROLE, "0")),
        (ROLES, (*BY_ROLE, "1")),
        ((*rows, "0"), (*BY_ROLE, "0")),
        ((*rows, "1"), (*BY_ROLE, "1")),
        (("cells", "0x0", "n"), ("cells", "0x0", "table")),
    }


def test_no_released_count_is_below_the_minimum_cell(wide, binary):
    """Walking each file generically finds no integer in 1..4 that counts units."""
    assert small_counts(wide[1], DEFAULT_MIN_CELL) == []
    assert small_counts(binary[1], DEFAULT_MIN_CELL) == []


def test_safe_build_is_idempotent(wide):
    """Releasing the same source twice gives the same file and leaves the source as it was."""
    source, _, _ = wide
    before = copy.deepcopy(source)
    first = safe_payload(source, DEFAULT_MIN_CELL)
    assert safe_payload(source, DEFAULT_MIN_CELL) == first
    assert source == before


def test_ordinary_run_never_builds_a_safe_source(wide_ped, tmp_path):
    """An ordinary run does not build a safe source: the release list is imported only under --safe-attempt."""
    assert run(wide_ped, tmp_path, *TWO, safe=False) is None
    assert "safe_attempt_min_cell" not in load(tmp_path)


def test_safe_meta_names_paths_by_base_name_only(wide_ped, tmp_path):
    """The directories of the input and of every path on the command line do not reach the safe file."""
    ped = tmp_path / "project-x" / "ped.tsv"
    ped.parent.mkdir()
    ped.write_bytes(wide_ped.read_bytes())
    out = tmp_path / "researcher-y" / "out"
    flags = ["--trait", "dx", "--permutations", "0", "--safe-attempt"]
    result = run_pedsum(["assortative-mating", "--in", str(ped), "--out", str(out), *flags])
    assert result.returncode == 0, result.stderr
    written = (out / "assortative_mating.txt").read_text()
    assert "project-x" not in written
    assert "researcher-y" not in written
    assert str(tmp_path) not in written
    meta = yaml.safe_load(written)
    assert meta["input"] == "ped.tsv"
    assert meta["command"] == f"pedsum assortative-mating --in ped.tsv --out out {' '.join(flags)}"


# ---------------------------------------------------------------------------
# Audit leak 7 and D7
# ---------------------------------------------------------------------------


def test_leak_7_no_largest_mate_network_share(wide):
    """The ordinary run gives the largest network's share of the couples, per sample and per cell; the safe run none."""
    _, written, ordinary = wide
    pairs = ordinary[AM]["mating_pairs"]
    assert 0 < pairs["largest_mate_network_share"] < 1
    assert round(pairs["largest_mate_network_share"] * pairs["n_total"]) > 1, "the share is one network's size"
    assert all("largest_mate_network_share" in cell for cell in ordinary[AM]["mate_correlation"])

    assert cell_record(written, "score", "score")["crude"]["pearson"]["r"] is not None, "the estimate is released"
    names = {key for path, _ in leaves(written) for key in path}
    assert "largest_mate_network_share" not in names
    extrema = {name for name in names if name in ("min", "max") or name.startswith(("max_", "min_", "largest"))}
    assert extrema == {"min_stratum_networks"}, "an option of the run, not an extremum of the data"


def _keys(node: Any) -> set[str]:
    """Every mapping key of a payload, at any depth."""
    if isinstance(node, dict):
        return set(node) | {key for value in node.values() for key in _keys(value)}
    if isinstance(node, list):
        return {key for value in node for key in _keys(value)}
    return set()


def test_d7_and_p2_no_complement_and_no_numeric_level(wide_ped, wide, tmp_path):
    """With every estimate released the file has no dropped-couple count, no second network count and no data value."""
    _, written, ordinary = wide
    dropped = {
        *("largest_mate_network_share", "n_dropped", "unknown_stratum", "mother_missing", "father_missing"),
        *("both_missing", "small_stratum", "degenerate_stratum", "failure_reasons", "n_values", "n_missing"),
    }
    assert dropped <= _keys(ordinary)
    assert not dropped & _keys(written)
    assert "n_mate_networks" in ordinary[AM]["mating_pairs"]
    assert "n_mate_networks" not in written[AM]["mating_pairs"]

    run(wide_ped, tmp_path / "plain", *ORDINAL, safe=False)
    run(wide_ped, tmp_path / "safe", *ORDINAL)
    plain_traits, safe_traits = load(tmp_path / "plain")[AM]["traits"], load(tmp_path / "safe")[AM]["traits"]
    assert [trait.get("levels") for trait in plain_traits] == [["0", "1", "2"], ["no", "yes"]]
    assert [trait.get("levels") for trait in safe_traits] == [None, ["no", "yes"]]
    assert "levels" not in safe_traits[0]
    assert WORD_LEVELS == ordinary_module._BINARY_WORDS, "every level classify_trait does not read from the data"


def test_released_estimates_are_the_ordinary_ones(wide, binary):
    """A released record holds the ordinary run's numbers, and a released count the ordinary count."""
    for _, written, ordinary in (wide, binary):
        for theirs in ordinary[AM]["mate_correlation"]:
            ours = cell_record(written, theirs["mother"], theirs["father"])
            assert (ours["n"], ours["n_mate_networks"]) == (theirs["n"], theirs["n_mate_networks"])
            for form in ("crude", "stratified"):
                for name, record in theirs.get(form, {}).items():
                    if name == "table":
                        assert [list(row.values()) for row in ours[form]["table"]["cells"].values()] == record
                        continue
                    released = ours[form][name]
                    assert "reason" not in released, (form, name)
                    for key in (VALUE_KEYS[name], "se", "ci", "p_perm", "boundary", "bootstrap"):
                        if key == "bootstrap" and key in record:
                            assert released[key] == {k: record[key][k] for k in ("requested", "valid", "failed")}
                        elif key in record:
                            assert released[key] == record[key], (form, name, key)
                    for sex in ("mothers", "fathers"):
                        if form == "stratified":
                            assert released[f"n_strata_{sex}"] == record[f"n_strata_{sex}"]
        for key in ("n_total", "n_mothers_multiple_mates", "n_fathers_multiple_mates"):
            assert written[AM]["mating_pairs"][key] == ordinary[AM]["mating_pairs"][key]
        for sex, theirs in ordinary[AM].get("within_person", {}).items():
            assert written[AM]["within_person"][sex] == theirs
    fixed = [
        cell["crude"]["tetrachoric"]["permutations"]["n_fixed_fathers"] for cell in binary[2][AM]["mate_correlation"]
    ]
    assert [cell["n_fixed_fathers"] for cell in binary[1][AM]["mate_correlation"]] == fixed
    assert "n_fixed_fathers" not in str(binary[1][AM]["mate_correlation"][0]["crude"])


# ---------------------------------------------------------------------------
# Supports
# ---------------------------------------------------------------------------


def _couples_of(rows: list[dict]) -> set[tuple[int, int]]:
    return {(row["mother"], row["father"]) for row in rows if row["mother"] != -1 and row["father"] != -1}


def test_supports_agree_with_a_brute_force_count(binary):
    """Each support is the smaller of the distinct mothers and fathers of its couples, counted directly."""
    source, _, _ = binary
    rows = mating_rows()
    dx = {row["id"]: ("no", "yes").index(row["dx"]) for row in rows}
    couples = _couples_of(rows)
    assert source["mating_pairs"]["n_total"] == len(couples)
    cell, gates = source["cells"]["0x0"], source["gates"]["cells"]["0x0"]
    for m_level in (0, 1):
        for f_level in (0, 1):
            members = [(m, f) for m, f in couples if (dx[m], dx[f]) == (m_level, f_level)]
            assert cell["table"][str(m_level)][str(f_level)] == len(members)
            support = min(len({m for m, _ in members}), len({f for _, f in members}))
            assert gates["table_support"][str(m_level)][str(f_level)] == support
    distinct = min(len({m for m, _ in couples}), len({f for _, f in couples}))
    assert gates["n_floor"] == source["gates"]["mating_pairs_floor"] == distinct < len(couples)
    assert at(source, GROUPS) == {"1": len(couples)}
    for sex, position in (("mother", 0), ("father", 1)):
        mates: dict[int, int] = {}
        for couple in couples:
            mates[couple[position]] = mates.get(couple[position], 0) + 1
        histogram = {str(k): list(mates.values()).count(k) for k in sorted(set(mates.values()))}
        assert source["gates"][f"mates_per_{sex}"] == histogram


def test_a_count_that_differs_from_pg_phenotype_writes_no_file(wide_ped, tmp_path):
    """When the couples counted for the supports are not the ones the estimates are over, the run fails."""

    def one_more_couple(payload: dict[str, Any]) -> None:
        payload["mate_correlation"][0]["n"] += 1

    assert run(wide_ped, tmp_path, *TWO, records=one_more_couple, code=1) is None
    assert list(tmp_path.iterdir()) == []


def _binary_ped(tmp_path: Path, table: list[list[int]], extra: list[tuple[int, int, int, int]] = ()) -> Path:
    """A pedigree whose couples give ``table``; ``extra`` adds couples ``(mother, father, m level, f level)``."""
    couples, level = table_couples(table)
    for mother, father, m_level, f_level in extra:
        couples.append((mother, father))
        level[mother], level[father] = m_level, f_level
    tmp_path.mkdir(exist_ok=True)
    return write_ped(tmp_path / "ped.tsv", couple_rows(couples, a=level))


A = ("--trait", "a", "--permutations", "19")


def test_d9_three_couples_of_one_man_are_one_unit(tmp_path):
    """Three couples in one cell release it at minimum cell 3 when three men father them, and not when one does."""
    table = [[4, 3], [3, 0]]
    several_men = _binary_ped(tmp_path / "x", table, [(900 + i, 50_900 + i, 1, 1) for i in range(3)])
    one_man = _binary_ped(tmp_path / "y", table, [(900 + i, 50_900, 1, 1) for i in range(3)])
    for ped, released in ((several_men, True), (one_man, False)):
        out_dir = ped.parent / "out"
        source = run(ped, out_dir, *A, "--min-cell", "3")
        assert source is not None
        assert source["cells"]["0x0"]["table"] == {"0": {"0": 4, "1": 3}, "1": {"0": 3, "1": 3}}
        assert source["gates"]["cells"]["0x0"]["table_support"]["1"]["1"] == (3 if released else 1)
        crude = cell_record(load(out_dir), "a", "a")["crude"]
        if released:
            assert crude["table"]["status"] == "intact"
            assert crude["table"]["cells"]["1"]["1"] == 3
            assert crude["odds_ratio"]["value"] is not None
        else:
            assert crude["table"] == {"status": "blanked", "total": 13}
            assert all(crude[name] == {VALUE_KEYS[name]: None, "reason": WITHHELD} for name in crude if name != "table")
        assert pinned_protected(source, 3) == []


def test_d6_a_zero_in_a_small_row_withholds_the_table_and_a_zero_elsewhere_stays(tmp_path):
    """A 2x2 with a zero in a row of two couples is withheld; a zero whose row and column are large is shown."""
    small_row = _binary_ped(tmp_path / "x", [[20, 20], [2, 0]])
    source = run(small_row, small_row.parent / "out", *A, "--min-cell", "3")
    assert source is not None
    assert cell_record(load(small_row.parent / "out"), "a", "a")["crude"]["table"] == {"status": "blanked", "total": 42}

    large = _binary_ped(tmp_path / "y", [[20, 20], [20, 0]])
    source = run(large, large.parent / "out", *A)
    assert source is not None
    crude = cell_record(load(large.parent / "out"), "a", "a")["crude"]
    assert crude["table"]["cells"] == {"0": {"0": 20, "1": 20}, "1": {"0": 20, "1": 0}}
    assert crude["odds_ratio"]["value"] == 0


# ---------------------------------------------------------------------------
# The estimates and counts that invert their inputs
# ---------------------------------------------------------------------------


def test_odds_ratio_does_not_invert_a_withheld_table(tmp_path):
    """Blanking a 2x2 to its margins beside its odds ratio gives every cell back; the safe run releases neither."""
    a, b, c, d = 2, 40, 30, 50
    ped = _binary_ped(tmp_path, [[a, b], [c, d]])
    run(ped, tmp_path / "plain", *A, safe=False)
    ordinary = cell_record(load(tmp_path / "plain"), "a", "a")["crude"]
    assert ordinary["table"] == [[a, b], [c, d]]
    odds_ratio = ordinary["odds_ratio"]["value"]
    assert odds_ratio == pytest.approx(a * d / (b * c), abs=half_width(odds_ratio))

    def naive_system() -> System:
        system = System()
        for cell in "abcd":
            system.var(cell)
        for pair, margin in (("ab", a + b), ("cd", c + d), ("ac", a + c), ("bd", b + d)):
            system.add(dict.fromkeys(pair, 1), margin, margin)
        return system

    assert naive_system().pinned("abcd") == [], "the margins alone pin nothing"
    naive = naive_system()
    add_nonlinear(
        naive,
        ["a"],
        [(first,) for first in range(min(a + b, a + c) + 1)],
        lambda first: (
            (a + b - first) * (a + c - first) > 0
            and abs(first * (d - a + first) / ((a + b - first) * (a + c - first)) - odds_ratio)
            <= half_width(odds_ratio)
        ),
    )
    assert naive.pinned("abcd") == ["a", "b", "c", "d"]
    assert naive.value_range("a") == (a, a)

    source = run(ped, tmp_path / "safe", *A)
    assert source is not None
    crude = cell_record(load(tmp_path / "safe"), "a", "a")["crude"]
    assert crude == {
        "table": {"status": "blanked", "total": a + b + c + d},
        "tetrachoric": {"rho": None, "reason": WITHHELD},
        "odds_ratio": {"value": None, "reason": WITHHELD},
        "phi": {"r": None, "reason": WITHHELD},
    }
    system, targets = reader_system(source, DEFAULT_MIN_CELL)
    cells = every_cell(source, [("cells", "0x0", "table")])
    assert (("cells", "0x0", "table"), ("0", "0")) in targets
    assert system.pinned(cells) == []
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []


def test_oracle_catches_an_odds_ratio_released_past_its_table(tmp_path):
    """With the table withheld, its margins known and the odds ratio released anyway, the reader has every cell.

    The margins are those a second cell over the same couples would show.
    The release list leaves them without the odds ratio, which pins nothing.
    """
    table = [[2, 40], [30, 50]]
    source = run(_binary_ped(tmp_path, table), tmp_path / "safe", *A)
    assert source is not None
    spec = assortative_mating_spec(source)
    path = ("cells", "0x0", "table")
    leaky = ReleaseSpec(
        tuple(
            Derived(rule.path, rule.reason, sources=(("cells", "0x0", "n"),)) if rule.path[-1] == "odds_ratio" else rule
            for rule in spec.rules
        ),
        spec.links,
    )
    cells = every_cell(source, [path])

    def pinned_with_margins(release_list: ReleaseSpec) -> list:
        system, _ = reader_system(source, DEFAULT_MIN_CELL, release_list)
        for i in ("0", "1"):
            system.add({(path, (i, "0")): 1, (path, (i, "1")): 1}, sum(table[int(i)]), sum(table[int(i)]))
            column = table[0][int(i)] + table[1][int(i)]
            system.add({(path, ("0", i)): 1, (path, ("1", i)): 1}, column, column)
        return system.pinned(cells)

    assert pinned_with_margins(spec) == []
    assert pinned_with_margins(leaky) == cells


def _two_trait_ped(tmp_path: Path, both: int, father_lacks: int, mother_lacks: int, neither: int) -> Path:
    """Couples by which mate has trait ``b``; every mate has trait ``a``. Levels alternate, so no table is small."""
    couples = [(i + 1, 50_001 + i) for i in range(both + father_lacks + mother_lacks + neither)]
    a = {person: (i // 2) % 2 for i, couple in enumerate(couples) for person in couple}
    a |= {father: (i // 4) % 2 for i, (_, father) in enumerate(couples)}
    b = dict(a)
    for i, (mother, father) in enumerate(couples):
        if both <= i < both + father_lacks or i >= both + father_lacks + mother_lacks:
            del b[father]
        if i >= both + father_lacks:
            del b[mother]
    tmp_path.mkdir(exist_ok=True)
    return write_ped(tmp_path / "ped.tsv", couple_rows(couples, a=a, b=b))


AB = ("--trait", "a", "b", "--permutations", "19")


def test_cell_counts_do_not_give_back_a_small_group_of_couples(tmp_path):
    """Four cells' counts give the couples in which neither mate has a trait, by inclusion and exclusion.

    With trait ``a`` complete, the couples total less the two mixed cells
    plus the cell of trait ``b`` is the number of couples lacking ``b`` on
    both sides. Two such couples withhold every count; twelve withhold none.
    """
    ped = _two_trait_ped(tmp_path / "x", both=40, father_lacks=12, mother_lacks=12, neither=2)
    run(ped, ped.parent / "plain", *AB, safe=False)
    ordinary = load(ped.parent / "plain")[AM]
    n = {(cell["mother"], cell["father"]): cell["n"] for cell in ordinary["mate_correlation"]}
    total = ordinary["mating_pairs"]["n_total"]
    assert n == {("a", "a"): 66, ("a", "b"): 52, ("b", "a"): 52, ("b", "b"): 40}
    assert total - n["a", "b"] - n["b", "a"] + n["b", "b"] == 2

    source = run(ped, ped.parent / "safe", *AB)
    assert source is not None
    assert at(source, GROUPS) == {"1": 2, "3": 12, "5": 12, "15": 40}
    written = load(ped.parent / "safe")[AM]
    assert written["mating_pairs"]["n_total"] is None
    for cell in written["mate_correlation"]:
        assert cell["n"] is None
        tables = {name: record for name, record in cell["crude"].items() if name == "table"}
        assert all(table == {"status": "blanked", "total": None} for table in tables.values())
        assert all(record["reason"] == WITHHELD for name, record in cell["crude"].items() if name != "table")
    assert small_counts(load(ped.parent / "safe"), DEFAULT_MIN_CELL) == []
    system, targets = reader_system(source, DEFAULT_MIN_CELL)
    assert ("group", "1") in targets
    assert system.value_range(("group", "1"))[1] > 2
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []

    # The same file with the counts released anyway: the group of two is pinned.
    spec = assortative_mating_spec(source)
    leaky = ReleaseSpec(
        tuple(Count(rule.path, rule.reason) if isinstance(rule, Count) else rule for rule in spec.rules), spec.links
    )
    assert ("group", "1") in pinned_protected(source, DEFAULT_MIN_CELL, leaky)

    # Sixteen mothers lacking ``b``, so that their mates hold each level of it eight times.
    ped = _two_trait_ped(tmp_path / "y", both=40, father_lacks=12, mother_lacks=16, neither=12)
    source = run(ped, ped.parent / "safe", *AB)
    assert source is not None
    written = load(ped.parent / "safe")[AM]
    assert written["mating_pairs"]["n_total"] == 80
    assert [cell["n"] for cell in written["mate_correlation"]] == [80, 56, 52, 40]
    assert all(cell["crude"]["tetrachoric"]["rho"] is not None for cell in written["mate_correlation"])
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []


def test_couples_no_cell_analyses_withhold_the_total_and_keep_the_cell(tmp_path):
    """With one trait, two couples with a missing value withhold the couples total; the cell's own count stays."""
    couples, level = table_couples([[10, 10], [10, 10]])
    couples += [(801, 50_801), (802, 50_802)]
    level |= {801: 0, 802: 1}
    ped = write_ped(tmp_path / "ped.tsv", couple_rows(couples, a=level))
    source = run(ped, tmp_path / "safe", *A)
    assert source is not None
    assert at(source, GROUPS) == {"0": 2, "1": 40}
    written = load(tmp_path / "safe")[AM]
    assert written["mating_pairs"]["n_total"] is None
    assert cell_record(load(tmp_path / "safe"), "a", "a")["n"] == 40
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []


def _without_support(spec: ReleaseSpec, *paths: tuple[str, ...]) -> ReleaseSpec:
    """``spec`` with the rules at ``paths`` gated on their own counts alone."""
    rules = tuple(
        type(rule)(**{name: getattr(rule, name) for name in rule.__dataclass_fields__} | {"support": None})
        if rule.path in paths
        else rule
        for rule in spec.rules
    )
    return ReleaseSpec(rules, spec.links)


def test_a1_network_counts_do_not_give_back_withheld_couple_counts(tmp_path):
    """Nobody remates, so a cell's Mate Networks are its couples: released, they are the withheld counts.

    Two mothers lack trait ``b``. Their couples are a group of two, which
    withholds every count of couples; the networks of the cells with and
    without them would differ by those two.
    """
    ped = _two_trait_ped(tmp_path, both=20, father_lacks=0, mother_lacks=2, neither=0)
    source = run(ped, tmp_path / "safe", *AB)
    assert source is not None
    assert at(source, GROUPS) == {"3": 2, "15": 20}
    assert [cell["n_mate_networks"] for cell in source["cells"].values()] == [22, 22, 20, 20]
    written = load(tmp_path / "safe")[AM]
    assert written["mating_pairs"] == {"n_total": None, "n_mothers_multiple_mates": 0, "n_fathers_multiple_mates": 0}
    assert [cell["n"] for cell in written["mate_correlation"]] == [None] * 4
    assert [cell["n_mate_networks"] for cell in written["mate_correlation"]] == [None] * 4
    # Every father has both traits and none remates, so the fathers with both are the couples.
    assert {sex: record["n"] for sex, record in written["within_person"].items()} == {"mothers": None, "fathers": None}
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []

    spec = assortative_mating_spec(source)
    leaky = _without_support(spec, *(("cells", key, "n_mate_networks") for key in source["cells"]))
    assert ("group", "3") in pinned_protected(source, DEFAULT_MIN_CELL, leaky)


@pytest.mark.parametrize(
    ("lacking", "n", "networks", "fathers"),
    [("b", [46, 46, 41, 41], [40, 40, 36, 36], None), ("ab", [41, 41, 41, 41], [36, 36, 36, 36], 40)],
)
def test_network_counts_do_not_count_the_rematings_of_a_group(tmp_path, lacking, n, networks, fathers):
    """Six men remate, and one of the second mates is among five mothers who lack a trait.

    The couples less the networks are six in a cell that analyses those
    mothers and five in one that does not, and the fathers' six extra mates
    are known from the couples total and the distinct fathers. Either
    difference says that one of the five couples is a remating.

    A cell that analyses every couple holds every extra mate, so its
    networks are the distinct fathers: the fathers with both traits are
    withheld with them. A cell without the five couples has networks the
    fathers only bound.
    """
    couples, level = table_couples([[10, 10], [10, 10]])
    for i in range(6):
        mother, (_, father) = 700 + i, couples[i]
        couples.append((mother, father))
        level[mother] = i % 2
    partial = {person: value for person, value in level.items() if person not in (700, 36, 37, 38, 39)}
    columns = {"a": partial if "a" in lacking else level, "b": partial}
    ped = write_ped(tmp_path / "ped.tsv", couple_rows(couples, children=True, **columns))
    source = run(ped, tmp_path / "safe", *AB)
    assert source is not None
    assert [cell["n_mate_networks"] for cell in source["cells"].values()] == networks
    written = load(tmp_path / "safe")[AM]
    assert written["mating_pairs"]["n_total"] == 46
    assert written["within_person"]["fathers"]["n"] == fathers
    assert [cell["n"] for cell in written["mate_correlation"]] == n
    assert [cell["n_mate_networks"] for cell in written["mate_correlation"]] == [None] * 4
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []


BIRTH_YEAR_STRATA = (
    *("--stratify-by", "birth_year", "--birth-year-col", "birth_year"),
    *("--birth-year-bin", "30", "--min-stratum-networks", "5"),
)


@pytest.mark.parametrize(("only_dropped", "released"), [(2, False), (5, True)])
def test_network_counts_do_not_count_the_mothers_of_dropped_couples_alone(tmp_path, only_dropped, released):
    """Forty analysed couples never remate; the couples of one stratum of fathers are dropped for a constant trait.

    Five of the dropped couples' mothers are also in an analysed couple and
    ``only_dropped`` are in none. No trait is missing, so the mothers with
    both traits are every mother, and less the networks of a cell whose
    couples never remate they are the mothers of dropped couples alone.
    """
    couples, level = table_couples([[10, 10], [10, 10]])
    dropped = [(couples[i][0], 60_000 + i) for i in range(5)] + [(800 + i, 61_000 + i) for i in range(only_dropped)]
    level |= {800 + i: i % 2 for i in range(only_dropped)} | {father: 0 for _, father in dropped}
    b = {person: value if person >= 60_000 else (person // 3) % 2 for person, value in level.items()}
    rows = couple_rows(couples + dropped, children=True, a=level, b=b)
    for row in rows:
        if 60_000 <= row["id"] < 100_000:
            row["birth_year"] = 1900
    source = run(write_ped(tmp_path / "ped.tsv", rows), tmp_path / "safe", *AB, *BIRTH_YEAR_STRATA)
    assert source is not None
    assert at(source, GROUPS) == {"0": 5 + only_dropped, "15": 40}
    assert [cell["n_mate_networks"] for cell in source["cells"].values()] == [40] * 4
    written = load(tmp_path / "safe")[AM]
    assert written["mating_pairs"]["n_total"] == 45 + only_dropped
    assert written["mating_pairs"]["n_mothers_multiple_mates"] == 5
    assert {sex: record["n"] for sex, record in written["within_person"].items()} == {
        "mothers": 40 + only_dropped,
        "fathers": 45 + only_dropped,
    }
    assert [cell["n"] for cell in written["mate_correlation"]] == [40] * 4
    assert [cell["n_mate_networks"] for cell in written["mate_correlation"]] == [40 if released else None] * 4
    assert all(cell["crude"]["tetrachoric"]["rho"] is not None for cell in written["mate_correlation"])
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []


def test_a2_missing_rows_do_not_give_back_the_non_parents_who_lack_a_trait(tmp_path):
    """Five mothers and one child lack trait ``a``: the rows without it, less the parents without it, are that child.

    The cells' counts and the mothers with both traits say that five
    mothers lack ``a`` and no father does, so the whole-pedigree count of
    six is withheld.
    """
    couples = [(i + 1, 50_001 + i) for i in range(20)]
    b = {person: (i // 2) % 2 for i, couple in enumerate(couples) for person in couple}
    b |= {father: (i // 4) % 2 for i, (_, father) in enumerate(couples)}
    a = {person: value for person, value in b.items() if person not in range(16, 21)}
    rows = couple_rows(couples, children=True, a=a, b=b)
    rows[-1]["a"] = "NA"
    source = run(write_ped(tmp_path / "ped.tsv", rows), tmp_path / "safe", *AB)
    assert source is not None
    written = load(tmp_path / "safe")[AM]
    assert written["mating_pairs"] == {"n_total": 20, "n_mothers_multiple_mates": 0, "n_fathers_multiple_mates": 0}
    assert {sex: record["n"] for sex, record in written["within_person"].items()} == {"mothers": 15, "fathers": 20}
    assert [trait["rows"] for trait in written["traits"]] == [
        {"status": "blanked", "total": 60},
        {"status": "intact", "total": 60, "bins": {"with_value": 60, "missing": 0}},
    ]
    assert at(source, (*BY_ROLE, "0")) == {
        "mothers": {"with_value": 15, "missing": 5},
        "fathers": {"with_value": 20, "missing": 0},
        "others": {"with_value": 19, "missing": 1},
    }
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []

    spec = assortative_mating_spec(source)
    leaky = _without_support(spec, ("supports", "trait_rows", "0"))
    assert ((*BY_ROLE, "0"), ("others", "missing")) in pinned_protected(source, DEFAULT_MIN_CELL, leaky)


def test_two_tables_do_not_give_back_a_small_level_of_a_group(tmp_path):
    """Two cells with the same mothers' trait share its margin over the couples both analyse.

    Six fathers lack trait ``b``; five of their mates are at one level of
    ``a`` and one at the other. The mothers' margins of the two tables
    would differ by five and one, so neither table is released.
    """
    couples, level = table_couples([[10, 10], [10, 10]])
    extra = [(900 + i, 50_900 + i) for i in range(6)]
    couples += extra
    level |= {mother: int(i == 5) for i, (mother, _) in enumerate(extra)}
    level |= {father: i % 2 for i, (_, father) in enumerate(extra)}
    b = {person: value for person, value in level.items() if person not in {father for _, father in extra}}
    source = run(write_ped(tmp_path / "ped.tsv", couple_rows(couples, a=level, b=b)), tmp_path / "safe", *AB)
    assert source is not None
    assert at(source, GROUPS) == {"5": 6, "15": 40}
    assert at(source, (*GROUP_LEVELS, "mother_0")) == {"1": {"0": 5, "1": 1}, "3": {"0": 20, "1": 20}}
    written = load(tmp_path / "safe")[AM]
    cells = {(cell["mother"], cell["father"]): cell for cell in written["mate_correlation"]}
    assert [cell["n"] for cell in cells.values()] == [46, 40, 46, 40]
    for key in (("a", "a"), ("a", "b")):
        assert cells[key]["crude"]["table"] == {"status": "blanked", "total": None}
        assert cells[key]["crude"]["tetrachoric"] == {"rho": None, "reason": WITHHELD}
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []

    spec = assortative_mating_spec(source)
    leaky = _without_support(spec, *(("cells", key, "table") for key in source["cells"]))
    assert ("group level", "1", "mother_0", "1") in pinned_protected(source, DEFAULT_MIN_CELL, leaky)


def _remating_ped(tmp_path: Path, remating_men: int, traits: int = 1) -> Path:
    """Forty couples of distinct people, then ``remating_men`` of the men each with a second mate."""
    couples, level = table_couples([[10, 10], [10, 10]])
    for i in range(remating_men):
        mother, (_, father) = 700 + i, couples[i]
        couples.append((mother, father))
        level[mother] = i % 2
    columns = {"a": level} | ({"b": {person: (person // 3) % 2 for person in level}} if traits == 2 else {})
    return write_ped(tmp_path / "ped.tsv", couple_rows(couples, children=True, **columns))


@pytest.mark.parametrize(("remating_men", "released"), [(0, True), (1, False), (4, False), (5, True)])
def test_few_rematings_withhold_the_network_and_person_counts(tmp_path, remating_men, released):
    """The couples less the networks, or less the distinct men, count the rematings: one to four are withheld."""
    ped = _remating_ped(tmp_path, remating_men, traits=2)
    source = run(ped, tmp_path / "safe", *AB)
    assert source is not None
    n = 40 + remating_men
    assert source["cells"]["0x0"]["n_mate_networks"] == 40
    assert source["within_person"]["fathers"]["n"] == 40
    assert source["gates"]["mates_per_father"] == (
        {"1": 40} if not remating_men else {"1": 40 - remating_men, "2": remating_men}
    )
    written = load(tmp_path / "safe")[AM]
    cell = written["mate_correlation"][0]
    assert cell["n"] == written["mating_pairs"]["n_total"] == n
    assert cell["n_mate_networks"] == (40 if released else None)
    assert written["mating_pairs"]["n_fathers_multiple_mates"] == (remating_men if released else None)
    assert written["mating_pairs"]["n_mothers_multiple_mates"] == 0
    fathers = written["within_person"]["fathers"]
    assert fathers["n"] == (40 if released else None)
    assert ("reason" not in fathers) == released
    if not released:
        assert fathers == {"estimator": "tetrachoric", "n": None, "rho": None, "reason": WITHHELD}
    assert written["within_person"]["mothers"]["n"] == n, "no woman remates, so the mothers are the couples"
    assert cell["crude"]["tetrachoric"]["rho"] is not None, "the estimates do not rest on the rematings"
    assert small_counts(load(tmp_path / "safe"), DEFAULT_MIN_CELL) == []
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []


def test_oracle_catches_a_person_count_released_beside_few_rematings(tmp_path):
    """One remating man: the couples total and the distinct fathers, released together, say that he is one."""
    source = run(_remating_ped(tmp_path, 1, traits=2), tmp_path / "safe", *AB)
    assert source is not None
    spec = assortative_mating_spec(source)
    n = ("within_person", "fathers", "n")
    leaky = ReleaseSpec(
        tuple(Count(rule.path, rule.reason) if rule.path == n else rule for rule in spec.rules), spec.links
    )
    # Every cell analyses every couple, so the distinct fathers are its withheld Mate Networks too.
    networks = [(("cells", key, "n_mate_networks"), None) for key in source["cells"]]
    assert pinned_protected(source, DEFAULT_MIN_CELL, leaky) == [*networks, ("fathers", 2)]


def _parent_counts(written: dict[str, Any]) -> list[Any]:
    """The counts of a file that say how many distinct mothers and fathers the couples have."""
    section = written[AM]
    return [
        *(section["mating_pairs"][f"n_{sex}_multiple_mates"] for sex in ("mothers", "fathers")),
        *(section["within_person"][sex]["n"] for sex in ("mothers", "fathers")),
        *(cell["n_mate_networks"] for cell in section["mate_correlation"]),
    ]


@pytest.mark.parametrize("leaves", [1, 4])
def test_few_individuals_in_no_couple_withhold_the_counts_of_distinct_parents(tmp_path, leaves):
    """Nobody remates and no trait is missing, so the individuals less twice the couples are those in no couple.

    The row total stays, as the other reports of the pedigree write it. The
    parents with several mates, the parents with both traits and the Mate
    Networks are withheld; each of them alone, beside the couples total,
    gives the distinct mothers and fathers.
    """
    rows = funnel_rows(leaves)
    source = run(write_ped(tmp_path / "ped.tsv", rows), tmp_path / "safe", *AB)
    assert source is not None
    assert at(source, ROLES) == {"mothers": 31, "fathers": 31, "others": leaves}
    written = load(tmp_path / "safe")
    assert written["n_total"] == len(rows) == 62 + leaves
    assert written[AM]["mating_pairs"]["n_total"] == 31
    assert _parent_counts(written) == [None] * 8
    assert all(record["reason"] == WITHHELD for record in written[AM]["within_person"].values())
    cells = written[AM]["mate_correlation"]
    assert [cell["n"] for cell in cells] == [31] * 4
    assert all(cell["crude"]["pearson"]["r"] is not None for cell in cells), "the estimates rest on the couples"
    assert small_counts(written, DEFAULT_MIN_CELL) == []
    system, targets = reader_system(source, DEFAULT_MIN_CELL)
    assert (ROLES, "others") in targets
    low, high = system.value_range((ROLES, "others"))
    assert low == leaves, "the couples bound the mothers and the fathers from above"
    assert high >= low + DEFAULT_MIN_CELL
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []

    spec = assortative_mating_spec(source)
    several = [("mating_pairs", f"n_{sex}_multiple_mates") for sex in ("mothers", "fathers")]
    without_roles = ReleaseSpec(
        tuple(
            Derived(rule.path, rule.reason, sources=rule.sources[:1]) if rule.path in several else rule
            for rule in spec.rules
        ),
        spec.links,
    )
    leaky = {
        "several mates": without_roles,
        "both traits": _without_support(spec, *(("within_person", sex, "n") for sex in ("mothers", "fathers"))),
        "networks": _without_support(spec, *(("cells", key, "n_mate_networks") for key in source["cells"])),
    }
    for route, stripped in leaky.items():
        assert (ROLES, "others") in pinned_protected(source, DEFAULT_MIN_CELL, stripped), route


def test_enough_individuals_in_no_couple_or_none_release_the_counts_of_distinct_parents(tmp_path):
    """Five individuals in no couple are a cell of the minimum size, and none is an empty cell: both release."""
    rows = funnel_rows(5)
    source = run(write_ped(tmp_path / "ped.tsv", rows), tmp_path / "safe", *AB)
    assert source is not None
    assert at(source, ROLES) == {"mothers": 31, "fathers": 31, "others": 5}
    released = [0, 0, 31, 31, 31, 31, 31, 31]
    assert _parent_counts(load(tmp_path / "safe")) == released
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []
    assert _parent_counts(safe_payload(source, 6)) == [None] * 8

    # No accepted pedigree has nobody outside its couples (the last child is a parent of nobody), so the empty
    # cell is the same source with those five rows taken out of every count of individuals.
    none = copy.deepcopy(source)
    none["n_total"] -= 5
    at(none, ROLES)["others"] = 0
    for index in none["traits"]:
        at(none, (*BY_ROLE, index))["others"]["with_value"] -= 5
        none["supports"]["trait_rows"][index]["with_value"] -= 5
    payload = safe_payload(none, DEFAULT_MIN_CELL)
    assert payload["n_total"] == 62
    assert _parent_counts(payload) == released
    system, targets = reader_system(none, DEFAULT_MIN_CELL)
    assert (ROLES, "others") not in targets
    assert system.value_range((ROLES, "others")) == (0, 0)


def test_a_father_alone_in_his_permutation_block_is_not_counted(tmp_path):
    """One father who lacks a trait is the only one of his missingness pattern: the ordinary run counts him."""
    ped = _two_trait_ped(tmp_path, both=40, father_lacks=1, mother_lacks=0, neither=0)
    run(ped, tmp_path / "plain", *AB, safe=False)
    ordinary = cell_record(load(tmp_path / "plain"), "a", "a")
    assert ordinary["crude"]["tetrachoric"]["permutations"]["n_fixed_fathers"] == 1
    source = run(ped, tmp_path / "safe", *AB)
    assert source is not None
    assert source["cells"]["0x0"]["n_fixed_fathers"] == 1
    written = load(tmp_path / "safe")
    assert cell_record(written, "a", "a")["n_fixed_fathers"] is None
    assert small_counts(written, DEFAULT_MIN_CELL) == []


# ---------------------------------------------------------------------------
# Strata and the options that change which couples count
# ---------------------------------------------------------------------------


def test_a_small_stratum_withholds_the_stratified_estimate_and_keeps_the_crude(tmp_path):
    """Four founding mothers are a stratum of four: the stratified estimate and the strata count are withheld."""
    ped = write_ped(tmp_path / "ped.tsv", mating_rows(seed=5, founders=8, per_generation=80, generations=2))
    flags = ("--trait", "score", "--permutations", "19", "--stratify-by", "depth", "--min-stratum-networks", "1")
    run(ped, tmp_path / "plain", *flags, safe=False)
    ordinary = cell_record(load(tmp_path / "plain"), "score", "score")
    assert ordinary["stratified"]["pearson"]["r"] is not None
    source = run(ped, tmp_path / "safe", *flags)
    assert source is not None
    gates = source["gates"]["cells"]["0x0"]
    assert min(gates["mother_strata"].values()) == 4
    cell = cell_record(load(tmp_path / "safe"), "score", "score")
    assert cell["stratified"] == {
        "pearson": {"r": None, "reason": WITHHELD, "n_strata_mothers": None, "n_strata_fathers": None}
    }
    assert cell["crude"]["pearson"]["r"] == ordinary["crude"]["pearson"]["r"]
    released = cell_record(safe_payload(source, 3), "score", "score")["stratified"]["pearson"]
    assert released["r"] == pytest.approx(ordinary["stratified"]["pearson"]["r"], abs=half_width(released["r"]))
    assert released["n_strata_mothers"] == 2


@pytest.mark.parametrize("gate", ["mother_strata", "father_strata", "mother_stratum_levels", "father_stratum_levels"])
def test_one_small_stratum_gate_withholds_only_the_stratified_estimate(wide, gate):
    """Two people in one stratum, or at one level of one stratum, withhold the stratified estimate of that cell."""
    source, written, _ = wide
    assert cell_record(written, "dx", "dx")["stratified"]["tetrachoric"]["rho"] is not None
    edited = copy.deepcopy(source)
    table = edited["gates"]["cells"]["0x0"][gate]
    first = next(iter(table))
    if isinstance(table[first], dict):
        table[first]["0"] = 2
    else:
        table[first] = 2
    payload = safe_payload(edited, DEFAULT_MIN_CELL)
    cell = cell_record(payload, "dx", "dx")
    assert cell["stratified"]["tetrachoric"] == {
        "rho": None,
        "reason": WITHHELD,
        "n_strata_mothers": None,
        "n_strata_fathers": None,
    }
    assert cell["crude"]["tetrachoric"]["rho"] is not None
    assert cell_record(payload, "score", "score")["stratified"]["pearson"]["r"] is not None


@pytest.mark.parametrize(
    ("path", "withheld", "kept"),
    [
        (("gates", "cells", "0x1", "levels", "1"), [("dx", "score")], [("dx", "dx"), ("score", "dx")]),
        (("gates", "cells", "0x0", "table_support", "1", "1"), [("dx", "dx")], [("dx", "score")]),
        (("gates", "cells", "1x1", "n_floor"), [("score", "score")], [("dx", "dx")]),
        (("gates", "cells", "1x0", "networks"), [("score", "dx")], [("dx", "dx"), ("dx", "score")]),
    ],
)
def test_one_small_support_withholds_the_estimates_that_rest_on_it(wide, path, withheld, kept):
    """Two people behind a level, a table cell or a cell's couples, or two networks, withhold that cell's estimates only."""
    source, _, _ = wide
    edited = copy.deepcopy(source)
    node = edited
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = 2
    payload = safe_payload(edited, DEFAULT_MIN_CELL)
    for mother, father in withheld:
        cell = cell_record(payload, mother, father)
        estimates = [
            record for form in ("crude", "stratified") for name, record in cell[form].items() if name != "table"
        ]
        assert all(record["reason"] == WITHHELD for record in estimates), (mother, father)
    for mother, father in kept:
        cell = cell_record(payload, mother, father)
        assert all("reason" not in record for name, record in cell["crude"].items() if name != "table")


def test_within_person_estimate_follows_its_own_table(wide, wide_ped, tmp_path):
    """Two mothers at one level, or at one pair of levels, withhold the mothers' within-person estimate only."""
    source, written, _ = wide
    assert written[AM]["within_person"]["mothers"]["rho"] is not None
    edited = copy.deepcopy(source)
    edited["gates"]["within_person"]["mothers"]["levels"]["1"] = 2
    within = safe_payload(edited, DEFAULT_MIN_CELL)[AM]["within_person"]
    n = source["within_person"]["mothers"]["n"]
    assert within["mothers"] == {"estimator": "biserial", "n": n, "rho": None, "reason": WITHHELD}
    assert within["fathers"]["rho"] is not None

    source = run(wide_ped, tmp_path, *ORDINAL)
    assert source is not None
    table = source["gates"]["within_person"]["mothers"]["table"]
    assert (len(table), table["0"]["1"]) == (3, 2), "two mothers of the lowest grade have the diagnosis"
    within = load(tmp_path)[AM]["within_person"]
    assert within["mothers"] == {"estimator": "polychoric", "n": 200, "rho": None, "reason": WITHHELD}
    table["0"]["1"] = 20
    within = safe_payload(source, DEFAULT_MIN_CELL)[AM]["within_person"]
    assert within["mothers"]["rho"] is not None
    assert within["fathers"]["reason"] == WITHHELD


def test_few_individuals_without_a_trait_withhold_the_trait_rows(tmp_path):
    """Two individuals lacking a trait are a small class: the rows table keeps its total and the estimates stay."""
    rows = mating_rows()
    for row in rows[-2:]:
        row["score"] = "NA"
    source = run(write_ped(tmp_path / "ped.tsv", rows), tmp_path / "out", "--trait", "score", "--permutations", "0")
    assert source is not None
    assert source["supports"]["trait_rows"]["0"] == {"with_value": len(rows) - 2, "missing": 2}
    written = load(tmp_path / "out")
    assert written[AM]["traits"][0]["rows"] == {"status": "blanked", "total": len(rows)}
    assert cell_record(written, "score", "score")["crude"]["pearson"]["r"] is not None
    assert small_counts(written, DEFAULT_MIN_CELL) == []
    assert pinned_protected(source, DEFAULT_MIN_CELL) == []


def test_options_change_the_couples_the_supports_count(wide_ped, tmp_path):
    """Strata, the stratum rule and the trait's type each change which couples a cell analyses, and its supports."""
    base = ("--trait", "grade", "--permutations", "0")
    sources = {
        name: run(wide_ped, tmp_path / name, *base, *flags)
        for name, flags in {
            "ordinal": ("--trait-type", "grade=ordinal"),
            "continuous": ("--trait-type", "grade=continuous"),
            "depth": ("--trait-type", "grade=ordinal", "--stratify-by", "depth", "--min-stratum-networks", "5"),
            "thin": ("--trait-type", "grade=ordinal", "--stratify-by", "depth", "--min-stratum-networks", "60"),
            "birth_year": (
                *("--trait-type", "grade=ordinal", "--stratify-by", "birth_year", "--birth-year-col", "birth_year"),
                *("--birth-year-bin", "30", "--min-stratum-networks", "5"),
            ),
        }.items()
    }
    assert all(source is not None for source in sources.values())
    ordinal, continuous = sources["ordinal"], sources["continuous"]
    assert len(ordinal["cells"]["0x0"]["table"]) == 3
    assert "table" not in continuous["cells"]["0x0"]
    assert list(continuous["cells"]["0x0"]["crude"]) == ["pearson", "spearman"]
    assert list(ordinal["cells"]["0x0"]["crude"]) == ["polychoric"]
    assert set(sources["depth"]["gates"]["cells"]["0x0"]) >= {"mother_strata", "mother_stratum_levels"}
    assert "mother_strata" not in ordinal["gates"]["cells"]["0x0"]
    assert sources["thin"]["cells"]["0x0"]["n"] < sources["depth"]["cells"]["0x0"]["n"] == ordinal["cells"]["0x0"]["n"]
    assert set(at(sources["depth"], GROUPS)) == {"1"}
    assert at(sources["thin"], GROUPS)["0"] > 0
    assert len(sources["birth_year"]["gates"]["cells"]["0x0"]["mother_strata"]) == 3
    for name in ("depth", "thin", "birth_year"):
        assert load(tmp_path / name)[AM]["settings"]["stratify_by"] == name.replace("thin", "depth")
    written = load(tmp_path / "ordinal")
    table = cell_record(written, "grade", "grade")["crude"]["table"]
    assert table["status"] == "intact"
    assert list(table["cells"]) == ["0", "1", "2"]


@pytest.mark.parametrize("value", ["2", "0", "-1", "3.5", "x"])
def test_min_cell_below_three_is_a_usage_error(wide_ped, tmp_path, value):
    """``--min-cell`` refuses anything but an integer of at least 3, and writes nothing."""
    result = run_pedsum(
        [
            *("assortative-mating", "--in", str(wide_ped), "--out", str(tmp_path / "out"), "--trait", "dx"),
            *("--safe-attempt", "--min-cell", value),
        ]
    )
    assert result.returncode == 2
    assert "--min-cell" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("min_cell", [3, 30, 201])
def test_min_cell_sets_the_gate_and_is_recorded(wide_ped, tmp_path, min_cell):
    """The file is that of the requested minimum cell, holds no count below it, and says which it was."""
    source = run(wide_ped, tmp_path, *TWO, *STRATA, "--min-cell", str(min_cell))
    assert source is not None
    written = load(tmp_path)
    assert written["safe_attempt_min_cell"] == min_cell
    expected = safe_payload(source, min_cell)
    assert written[AM]["mate_correlation"] == yaml.safe_load(yaml.safe_dump(_rounded(expected)))[AM]["mate_correlation"]
    assert small_counts(written, min_cell) == []
    tables = [cell["crude"]["table"]["status"] for cell in written[AM]["mate_correlation"] if "table" in cell["crude"]]
    assert tables == ["intact" if min_cell < 30 else "blanked"]
    if min_cell == 201:
        assert all(cell["n"] is None for cell in written[AM]["mate_correlation"])


def _rounded(payload: dict[str, Any]) -> dict[str, Any]:
    from pedsum.report import ASSORTATIVE_MATING_FIGURES, _round_floats

    return _round_floats(payload, figures=ASSORTATIVE_MATING_FIGURES)


# ---------------------------------------------------------------------------
# Exact oracle
# ---------------------------------------------------------------------------

TRAITS = {
    "dx": (),
    "grade": ("--trait-type", "grade=ordinal"),
    "score": (),
}


def _oracle_run(rows: list[dict], traits: tuple[str, ...], strata: int | None, tmp: Path) -> None:
    """At minimum cell 3 and 5: no protected number of the safe file is pinned, and it holds no small count."""
    ped = write_ped(tmp / "p.tsv", rows)
    flags = ["--trait", *traits, "--permutations", "9", *(flag for trait in traits for flag in TRAITS[trait])]
    if strata is not None:
        flags += ["--stratify-by", "depth", "--min-stratum-networks", str(strata)]
    for min_cell in (3, DEFAULT_MIN_CELL):
        out_dir = tmp / f"out{min_cell}"
        source = run(ped, out_dir, *flags, "--min-cell", str(min_cell), code=None)
        if source is None:
            # A trait that is constant or missing throughout is refused before anything is written.
            assert not out_dir.exists() or not list(out_dir.iterdir())
            return
        assert pinned_protected(source, min_cell) == []
        assert small_counts(load(out_dir), min_cell) == []


@pytest.mark.parametrize("min_cell", [3, DEFAULT_MIN_CELL, 10, 30])
def test_oracle_wide_pedigree(wide, binary, min_cell):
    """No protected number of either run's release is pinned, at any minimum cell."""
    assert pinned_protected(wide[0], min_cell) == []
    assert pinned_protected(binary[0], min_cell) == []


@pytest.mark.parametrize("traits", [("dx",), ("dx", "grade"), ("dx", "score"), ("grade", "score")])
@pytest.mark.parametrize("strata", [None, 1])
def test_oracle_adversarial_pedigree(tmp_path, traits, strata):
    """A small pedigree with missing values, rematings and tables of small cells, with and without strata."""
    rows = mating_rows(seed=2, founders=18, per_generation=26, generations=2, missing=0.12, remating=2)
    _oracle_run(rows, traits, strata, tmp_path)


@pytest.mark.slow
@property_budget(60)
@given(
    seed=st.integers(0, 10_000),
    founders=st.integers(4, 40),
    per_generation=st.integers(4, 60),
    generations=st.integers(1, 3),
    missing=st.sampled_from([0.0, 0.0, 0.03, 0.1, 0.3]),
    remating=st.integers(0, 6),
    traits=st.sampled_from([("dx",), ("score",), ("grade",), ("dx", "grade"), ("dx", "score"), ("grade", "score")]),
    strata=st.sampled_from([None, None, 1, 3]),
)
def test_oracle_generated_pedigrees(seed, founders, per_generation, generations, missing, remating, traits, strata):
    """On generated pedigrees no protected number is pinned and no file holds a small count, at minimum cell 3 and 5."""
    rows = mating_rows(seed, founders, per_generation, generations, missing, remating)
    with tempfile.TemporaryDirectory() as tmp:
        _oracle_run(rows, traits, strata, Path(tmp))


@given(
    st.lists(st.tuples(st.integers(0, 39), st.integers(0, 39)), max_size=30, unique=True),
    st.sampled_from((40, 10_000)),
)
def test_network_labels_partition_the_couples_as_union_find_does(pairs, n_rows):
    """Over a pedigree of any size, on either side of the table threshold, the labels are union-find's partition."""
    parent = list(range(40))

    def find(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for mother, father in pairs:
        parent[find(mother)] = find(father)
    expected = [find(mother) for mother, _ in pairs]

    couples = _Couples(
        np.array([m for m, _ in pairs], dtype=np.int64), np.array([f for _, f in pairs], dtype=np.int64), n_rows
    )
    labels, n_labels = couples.network_labels()
    assert labels.shape == (len(pairs),)
    assert n_labels == couples.n_networks() == len(set(expected))
    assert all(0 <= label < n_labels for label in labels.tolist())
    assert len(set(zip(labels.tolist(), expected, strict=True))) == n_labels, "the same partition of the couples"
