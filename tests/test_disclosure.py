"""The disclosure engine (``pedsum.disclosure``): merge, blank, taint, the release list, and the cases that broke v1.

Each regression case from the Codex review is shown twice through the exact
integer oracle (``disclosure_oracle``): pinned under naive cell blanking,
then not pinned under the engine.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from disclosure_oracle import (
    InconsistentSystemError,
    System,
    add_moments,
    add_nonlinear,
    close_to,
    system_from_release,
)
from hypothesis import given
from hypothesis import strategies as st

from pedsum.disclosure import (
    MIN_CELL_FLOOR,
    Contingency,
    ContingencyTable,
    Continuous,
    Count,
    Derived,
    FreqTable,
    Link,
    Plain,
    ProtectedCount,
    ReleaseSpec,
    Rule,
    Status,
    Table,
    build_release,
    protect,
    protect_count,
    protect_payload,
    release_derived,
    release_distribution,
    shared_total,
)
from pedsum.sections import numeric_distribution

# Reports are written at 4 decimals (``report._round_floats``).
HALF_WIDTH = 0.5e-4


def ordered(counts: dict[str, int], support: tuple[int, ...] | None = None) -> FreqTable:
    """An ordered table from a label-to-count mapping."""
    return FreqTable(tuple(counts.items()), ordered=True, support=support)


def released(table: FreqTable, min_cell: int) -> dict[str, int]:
    """The bins ``protect`` releases, as a mapping."""
    return {b.label: b.count for b in protect(table, min_cell).bins}


def value_counts(values: list[int]) -> dict[str, int]:
    """The full-resolution histogram of ``values``, one bin per integer from the smallest to the largest."""
    seen = Counter(values)
    return {str(v): seen[v] for v in range(min(values), max(values) + 1)}


# ---------------------------------------------------------------------------
# Merge and blank
# ---------------------------------------------------------------------------


def test_intact_table_is_released_as_is():
    """No small bin: every bin and the total are released."""
    out = protect(ordered({"1": 32, "2": 60, "3": 10}), 5)
    assert out.status is Status.INTACT
    assert {b.label: b.count for b in out.bins} == {"1": 32, "2": 60, "3": 10}
    assert out.total == 102


def test_upper_tail_merges_until_not_small():
    """A small top bin merges inward, twice here, and the label says what it covers."""
    out = protect(ordered({"1": 32, "2": 67, "3": 2, "4": 1}), 5)
    assert out.status is Status.MERGED
    assert {b.label: b.count for b in out.bins} == {"1": 32, "2+": 70}
    assert out.bins[-1].covers == ("2", "3", "4")
    assert out.total == 102


def test_upper_tail_stops_at_first_sufficient_merge():
    """Merging stops as soon as the merged bin reaches the minimum cell."""
    assert released(ordered({"1": 32, "2": 67, "3": 2, "4": 1}), 3) == {"1": 32, "2": 67, "3+": 3}


def test_lower_tail_merges():
    """A small bottom bin merges upward into a range label."""
    assert released(ordered({"0": 2, "1": 9, "2": 30}), 5) == {"0-1": 11, "2": 30}


def test_both_tails_merge():
    """Each end merges on its own."""
    assert released(ordered({"0": 1, "1": 9, "2": 30, "3": 8, "4": 2}), 5) == {"0-1": 10, "2": 30, "3+": 10}


def test_non_integer_labels_merge_to_a_span():
    """Labels that are not integers merge to ``first..last``."""
    table = ordered({"0": 50, "(0,0.06]": 9, "(0.06,0.12]": 2, "(0.12,0.25]": 1})
    assert released(table, 5) == {"0": 50, "(0,0.06]..(0.12,0.25]": 12}


def test_small_interior_bin_blanks_the_table():
    """A small bin between sufficient tails cannot merge: only the total is kept."""
    out = protect(ordered({"0": 50, "1": 30, "2": 2, "3": 9}), 5)
    assert out.status is Status.BLANKED
    assert out.bins == ()
    assert out.total == 91


def test_merging_to_one_bin_blanks_the_table():
    """Tails that meet leave one bin, which is the total: blanked."""
    out = protect(ordered({"1": 4, "2": 0, "3": 1}), 5)
    assert out.status is Status.BLANKED
    assert out.total == 5


def test_small_total_is_withheld():
    """A total in ``(0, min_cell)`` is not released either."""
    out = protect(ordered({"1": 2, "2": 1}), 5)
    assert out.status is Status.BLANKED
    assert out.total is None


def test_unordered_table_blanks_on_any_small_bin():
    """Unordered bins have no tail to merge."""
    out = protect(FreqTable((("FS", 40), ("MHS", 2), ("PO", 80)), ordered=False), 5)
    assert out.status is Status.BLANKED
    assert out.total == 122
    intact = protect(FreqTable((("FS", 40), ("MHS", 0), ("PO", 80)), ordered=False), 5)
    assert intact.status is Status.INTACT


def test_zero_bins_stay_visible_in_a_histogram():
    """D6: a zero bin of a 1-D table is released and triggers nothing, at either end or inside."""
    table = ordered({"0": 0, "1": 12, "2": 0, "3": 7, "4": 0})
    out = protect(table, 5)
    assert out.status is Status.INTACT
    assert released(table, 5) == {"0": 0, "1": 12, "2": 0, "3": 7, "4": 0}


def test_merge_absorbs_outer_zero_bins():
    """A tail merge starts at the outermost nonzero bin and takes the empty bins beyond it."""
    assert released(ordered({"1": 20, "2": 9, "3": 2, "4": 0}), 5) == {"1": 20, "2+": 11}


def test_contingency_zero_in_a_small_row_blanks():
    """D6: a zero in a row of total 2 is withheld with the table, because that row holds a small nonzero cell."""
    out = protect(ContingencyTable(("a", "b"), ("x", "y"), ((2, 0), (30, 40))), 3)
    assert out.status is Status.BLANKED
    assert out.counts is None
    assert out.total == 72


def test_contingency_zero_in_large_margins_is_released():
    """A zero whose row and column totals reach the minimum cell is not small."""
    out = protect(ContingencyTable(("a", "b"), ("x", "y"), ((10, 0), (30, 40))), 3)
    assert out.status is Status.INTACT
    assert out.counts == ((10, 0), (30, 40))


def test_contingency_small_cell_blanks():
    """Any nonzero cell below the minimum cell blanks every cell."""
    out = protect(ContingencyTable(("a", "b"), ("x", "y"), ((2, 40), (30, 50))), 5)
    assert out.status is Status.BLANKED
    assert out.render() == {"status": "blanked", "total": 122}


# ---------------------------------------------------------------------------
# Support gate (D9)
# ---------------------------------------------------------------------------


def test_one_sibship_of_three_does_not_release_its_pairs():
    """Three FS pairs from one family are one family: nothing is released at min_cell 3, not even the total."""
    out = protect(FreqTable((("FS", 3),), ordered=False, support=(1,)), 3)
    assert out.status is Status.BLANKED
    assert out.total is None
    assert protect_count(3, 3, support=1).value is None


def test_support_gates_instead_of_the_count():
    """A cell passes on its families, not its pairs."""
    table = FreqTable((("FS", 30), ("MHS", 12)), ordered=False, support=(10, 2))
    assert protect(table, 3).status is Status.BLANKED
    assert protect(FreqTable(table.bins, ordered=False, support=(10, 3)), 3).status is Status.INTACT
    assert protect_count(12, 3, support=3).value == 12


def test_count_below_its_support_is_gated_on_the_count():
    """A support larger than the count is impossible input; it does not lift a small count over the gate."""
    assert protect_count(2, 5, support=18).value is None
    assert protect(FreqTable((("FS", 2), ("PO", 90)), ordered=False, support=(18, 40)), 5).status is Status.BLANKED
    table = ContingencyTable(("a", "b"), ("x", "y"), ((2, 40), (30, 50)), support=((18, 35), (30, 44)))
    assert protect(table, 5).status is Status.BLANKED


def test_contingency_support_gate():
    """A contingency cell of many pairs from two families is small."""
    table = ContingencyTable(("a", "b"), ("x", "y"), ((9, 9), (9, 9)), support=((5, 5), (5, 2)))
    assert protect(table, 3).status is Status.BLANKED


# ---------------------------------------------------------------------------
# Scalars, taint, continuous stats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("count", "value"), [(0, 0), (1, None), (4, None), (5, 5), (900, 900)])
def test_scalar_count(count, value):
    """A scalar count is released when 0 or at least the minimum cell."""
    assert protect_count(count, 5).value == value


def test_derived_follows_its_sources():
    """P4: a derived value is released only if every source is intact."""
    intact = protect(ordered({"1": 32, "2": 60}), 5)
    merged = protect(ordered({"1": 32, "2": 67, "3": 2, "4": 1}), 5)
    blanked = protect_count(2, 5)
    assert release_derived(1.7, [intact]) == 1.7
    assert release_derived(1.7, [intact, merged]) is None
    assert release_derived(1.7, [intact, blanked]) is None


@pytest.mark.parametrize(
    ("n", "keys"),
    [
        (0, {"n"}),
        (4, set()),
        (5, {"n", "mean", "std"}),
        (19, {"n", "mean", "std"}),
        (20, {"n", "mean", "std", "q1", "median", "q3"}),
    ],
)
def test_continuous_thresholds(n, keys):
    """P5: mean and sd from ``min_cell`` values, quartiles from four times that, never the extrema or ``nz``."""
    dist = numeric_distribution(np.linspace(0.0, 1.0, n))
    assert dist.n == n
    assert set(release_distribution(dist, 5)) == keys


def test_integer_count_stats_follow_the_histogram():
    """P4 for a count variable: any merge or blank drops mean, sd and quartiles."""
    values = [1] * 32 + [2] * 67 + [3, 3, 4]
    dist = numeric_distribution(np.array(values))
    histogram = protect(ordered(value_counts(values)), 5)
    assert histogram.status is Status.MERGED
    assert release_distribution(dist, 5, histogram) == {}
    intact = protect(ordered({"1": 32, "2": 70}), 5)
    assert set(release_distribution(dist, 5, intact)) == {"mean", "std", "q1", "median", "q3"}


def test_distribution_n_is_not_written():
    """``Distribution.n`` gates the release and stays out of the report mapping."""
    dist = numeric_distribution(np.arange(7))
    assert dist.n == 7
    assert "n" not in dist.as_dict()


@pytest.mark.parametrize("min_cell", [2, 0, -1, 2.5, True])
def test_min_cell_below_three_raises(min_cell):
    """The engine refuses a minimum cell below 3 at every entry point."""
    assert MIN_CELL_FLOOR == 3
    spec = ReleaseSpec((Plain(("command",), "names the run"),))
    with pytest.raises(ValueError, match="min_cell"):
        protect(ordered({"1": 9}), min_cell)
    with pytest.raises(ValueError, match="min_cell"):
        protect_count(9, min_cell)
    with pytest.raises(ValueError, match="min_cell"):
        release_distribution(numeric_distribution(np.arange(9)), min_cell)
    with pytest.raises(ValueError, match="min_cell"):
        build_release({"command": "summarize"}, spec, min_cell)


def test_a_share_is_not_a_count():
    """A float where a count belongs is an error, not a value that slips under the gate."""
    with pytest.raises(TypeError, match="non-negative integers"):
        FreqTable((("1", 0.3), ("2", 0.7)), ordered=True)
    with pytest.raises(TypeError, match="non-negative integers"):
        protect_count(0.3, 5)


# ---------------------------------------------------------------------------
# Release list
# ---------------------------------------------------------------------------

SIBSHIP_VALUES = [1] * 32 + [2] * 67 + [3, 3, 4]

SPEC = ReleaseSpec(
    rules=(
        Plain(("command",), "names the run, not the data"),
        Plain(("min_cell",), "the threshold applied"),
        Count(("n_total",), "population size; released if 0 or at least min_cell"),
        Count(("pairs", "n_fs"), "pair count gated on contributing families", support=("pairs", "n_fs_families")),
        Table(("sibships", "size"), "sibship-size histogram, tails merged", ordered=True),
        Continuous(("sibships", "children"), "moments of the histogram above", histogram=("sibships", "size")),
        Table(("sex",), "count by sex", ordered=False),
        Contingency(("mates", "table"), "mate-pair 2x2", support=("mates", "families")),
        Derived(("mates", "odds_ratio"), "computed from the 2x2", sources=(("mates", "table"),)),
        Derived(("ne_sex_ratio",), "computed from the sex counts", sources=(("sex",),)),
        Continuous(("inbreeding", "F"), "continuous; n-gated moments"),
    ),
    links=shared_total("sex splits the whole population", ("n_total",), ("sex",)),
)


def payload() -> dict:
    """A payload exercising every rule kind, plus keys no rule names."""
    return {
        "command": "summarize",
        "min_cell": 5,
        "input": "/home/someone/pedigree.tsv",
        "n_total": 210,
        "pairs": {"n_fs": 3, "n_fs_families": 1, "ids": [101, 102, 103]},
        "sibships": {
            "size": value_counts(SIBSHIP_VALUES),
            "children": numeric_distribution(np.array(SIBSHIP_VALUES)),
            "largest": 4,
        },
        "sex": {"male": 100, "female": 110},
        "mates": {
            "table": {"a": {"x": 20, "y": 40}, "b": {"x": 30, "y": 50}},
            "families": {"a": {"x": 18, "y": 35}, "b": {"x": 30, "y": 44}},
            "odds_ratio": 0.8333,
        },
        "ne_sex_ratio": 209.5,
        "inbreeding": {"F": numeric_distribution(np.linspace(0, 0.25, 40)), "max_F": 0.25},
        "new_upstream_key": {"anything": 1},
    }


def test_release_holds_only_listed_leaves():
    """P1: the result is the listed leaves, protected; an unlisted key is absent and nothing raises."""
    source = payload()
    out = build_release(source, SPEC, 5)
    assert out == {
        "command": "summarize",
        "min_cell": 5,
        "n_total": 210,
        "pairs": {"n_fs": None},
        "sibships": {
            "size": {"status": "merged", "total": 102, "bins": {"1": 32, "2+": 70}},
            "children": None,
        },
        "sex": {"status": "intact", "total": 210, "bins": {"male": 100, "female": 110}},
        "mates": {
            "table": {
                "status": "intact",
                "total": 140,
                "cells": {"a": {"x": 20, "y": 40}, "b": {"x": 30, "y": 50}},
            },
            "odds_ratio": 0.8333,
        },
        "ne_sex_ratio": 209.5,
        "inbreeding": {"F": release_distribution(source["inbreeding"]["F"], 5)},
    }
    assert set(out["inbreeding"]["F"]) == {"n", "mean", "std", "q1", "median", "q3"}


def test_release_does_not_alias_or_change_the_payload():
    """The result is a new dict; the payload's tables are untouched."""
    source = payload()
    out = build_release(source, SPEC, 5)
    assert source["sibships"]["size"] == value_counts(SIBSHIP_VALUES)
    assert out["sex"]["bins"] is not source["sex"]


def test_missing_leaves_are_absent_and_orphaned_stats_withheld():
    """A leaf the payload lacks is skipped, one it holds as None is null, and a statistic with no source table is withheld."""
    out = build_release(
        {"command": "summarize", "mates": {"odds_ratio": 2.0}, "sex": None, "ne_sex_ratio": 9.5}, SPEC, 5
    )
    assert out == {"command": "summarize", "mates": {"odds_ratio": None}, "sex": None, "ne_sex_ratio": None}


def test_small_cell_taints_the_statistic():
    """P4 through the release list: a small 2x2 cell withholds the odds ratio."""
    source = payload()
    source["mates"]["table"]["a"]["x"] = 2
    source["mates"]["families"]["a"]["x"] = 2
    out = build_release(source, SPEC, 5)
    assert out["mates"] == {"table": {"status": "blanked", "total": 122}, "odds_ratio": None}


def test_release_is_idempotent_in_min_cell():
    """Two builds from one payload agree."""
    assert build_release(payload(), SPEC, 5) == build_release(payload(), SPEC, 5)


def test_every_rule_has_a_reason():
    """P1: a rule cannot be built without a written reason."""
    assert all(rule.reason.strip() for rule in SPEC.rules)
    for reason in ("", "   "):
        with pytest.raises(ValueError, match="no reason"):
            Plain(("command",), reason)
        with pytest.raises(ValueError, match="no reason"):
            Table(("sex",), reason, ordered=False)
    with pytest.raises(TypeError):
        Plain(("command",))


def test_spec_rejects_bad_declarations():
    """A spec refuses duplicate or nested paths and references to undeclared tables."""
    plain = Plain(("a",), "meta")
    with pytest.raises(ValueError, match="two rules"):
        ReleaseSpec((plain, Count(("a",), "count")))
    with pytest.raises(ValueError, match="parent"):
        ReleaseSpec((plain, Count(("a", "b"), "count")))
    with pytest.raises(ValueError, match="not a Count or Table or Contingency rule"):
        ReleaseSpec((plain, Derived(("d",), "stat", sources=(("a",),))))
    with pytest.raises(ValueError, match="not a Table rule"):
        ReleaseSpec((Count(("a",), "count"), Continuous(("c",), "stat", histogram=("a",))))
    with pytest.raises(ValueError, match="names no source"):
        Derived(("d",), "stat", sources=())
    with pytest.raises(ValueError, match="a link"):
        ReleaseSpec((plain,), links=(Link((("a",), None), ((("b",), None),), "same people"),))
    with pytest.raises(ValueError, match="non-empty tuple"):
        Rule((), "reason")


# ---------------------------------------------------------------------------
# The oracle itself
# ---------------------------------------------------------------------------


def test_oracle_flags_a_single_hidden_cell_under_a_total():
    """One blanked cell beside a total is the textbook pin (R2)."""
    system = System()
    for cell in "ab":
        system.var(cell)
    system.fix("a", 7)
    system.add({"a": 1, "b": 1}, 10, 10)
    assert system.pinned(["b"]) == ["b"]
    assert system.value_range("b") == (3, 3)


def test_oracle_passes_two_hidden_cells_under_a_total():
    """Two blanked cells under one total have many solutions."""
    system = System()
    for cell in "ab":
        system.var(cell)
    system.add({"a": 1, "b": 1}, 10, 10)
    assert system.pinned(["a", "b"]) == []
    assert system.value_range("a") == (0, 10)


def test_oracle_pins_through_a_bound():
    """A bound can pin: two hidden cells of at most 5 that sum to 10."""
    system = System()
    for cell in "ab":
        system.var(cell, 0, 5)
    system.add({"a": 1, "b": 1}, 10, 10)
    assert system.pinned(["a", "b"]) == ["a", "b"]


def test_oracle_rejects_an_inconsistent_system():
    """No solution is an error, never a silent pass."""
    system = System()
    system.var("a", 0, 5)
    system.add({"a": 1}, 7, 7)
    with pytest.raises(InconsistentSystemError):
        system.pinned(["a"])


def test_oracle_pins_through_a_declared_link():
    """Two blanked tables are free alone and pinned once a link ties a cell of one to an intact table."""
    spec = ReleaseSpec(
        (Table(("by_sex",), "by sex", ordered=False), Table(("by_role",), "by role", ordered=False)),
        links=(Link((("by_role",), "founder"), ((("by_sex",), "male"),), "every founder here is male"),),
    )
    protected = protect_payload({"by_sex": {"male": 3, "female": 7}, "by_role": {"founder": 3, "other": 7}}, spec, 5)
    assert all(table.status is Status.BLANKED for table in protected.values())
    system, hidden = system_from_release(spec, protected, 5)
    assert system.pinned(hidden) == []

    protected[("by_role",)] = protect(FreqTable((("founder", 3), ("other", 7)), ordered=False), 3)
    system, hidden = system_from_release(spec, protected, 5)
    assert sorted(system.pinned(hidden)) == [(("by_sex",), "female"), (("by_sex",), "male")]


# ---------------------------------------------------------------------------
# Codex counterexamples: pinned under naive cell blanking, not under the engine
# ---------------------------------------------------------------------------


def engine_pins(source: dict, spec: ReleaseSpec, min_cell: int) -> tuple[dict, list]:
    """The engine's release of ``source`` and the withheld cells the oracle can still pin from it."""
    system, hidden = system_from_release(spec, protect_payload(source, spec, min_cell), min_cell)
    assert hidden, "the case must withhold something"
    return build_release(source, spec, min_cell), system.pinned(hidden)


def test_cross_table_triangle():
    """Three overlapping tables, each blanked consistently, pin every cell jointly: ``x+y=10, x+z=11, y+z=15``."""
    for hidden_pair, total in ((("x", "y"), 10), (("x", "z"), 11), (("y", "z"), 15)):
        alone = System()
        for cell in hidden_pair:
            alone.var(cell)
        alone.add(dict.fromkeys(hidden_pair, 1), total, total)
        assert alone.pinned(hidden_pair) == [], "each table passes a per-table check"

    naive = System()
    for cell in "xyz":
        naive.var(cell)
    naive.add({"x": 1, "y": 1}, 10, 10)
    naive.add({"x": 1, "z": 1}, 11, 11)
    naive.add({"y": 1, "z": 1}, 15, 15)
    assert naive.pinned("xyz") == ["x", "y", "z"]
    assert [naive.value_range(cell) for cell in "xyz"] == [(3, 3), (7, 7), (8, 8)]

    spec = ReleaseSpec(
        (
            Table(("cells",), "each fact once (P2)", ordered=False),
            *(Derived((name,), "a sum of cells", sources=(("cells",),)) for name in ("xy", "xz", "yz")),
        )
    )
    out, pinned = engine_pins({"cells": {"x": 3, "y": 7, "z": 8}, "xy": 10, "xz": 11, "yz": 15}, spec, 5)
    assert out == {"cells": {"status": "blanked", "total": 18}, "xy": None, "xz": None, "yz": None}
    assert pinned == []


def test_triangle_listed_as_three_tables_is_caught_by_the_oracle():
    """The engine cannot see shared cells: a spec that lists the overlapping tables leaks, and the links expose it."""
    tables = {"a": ("x", "y"), "b": ("x", "z"), "c": ("y", "z")}
    spec = ReleaseSpec(
        tuple(Table((name,), "overlapping table", ordered=False) for name in tables),
        links=(
            Link((("a",), "x"), ((("b",), "x"),), "same cell"),
            Link((("a",), "y"), ((("c",), "y"),), "same cell"),
            Link((("b",), "z"), ((("c",), "z"),), "same cell"),
        ),
    )
    value = {"x": 3, "y": 7, "z": 8}
    source = {name: {cell: value[cell] for cell in cells} for name, cells in tables.items()}
    out, pinned = engine_pins(source, spec, 5)
    assert [out[name]["status"] for name in tables] == ["blanked", "blanked", "intact"]
    assert sorted(pinned) == [(("a",), "x"), (("a",), "y"), (("b",), "x"), (("b",), "z")]


def test_lp_relaxation_misses_the_integer_pin():
    """Values ``(1, 1, 1, 1, 3)``: the LP relaxation leaves every bin free, yet one integer histogram fits."""
    values = [1, 1, 1, 1, 3]
    dist = numeric_distribution(np.array(values))
    mean, sd = round(dist.mean, 4), round(dist.std, 4)

    naive = System()
    cells = {f"c{v}": v for v in range(1, 7)}
    for cell in cells:
        naive.var(cell)
    naive.add(dict.fromkeys(cells, 1), 5, 5)
    add_moments(naive, cells, 5, mean, sd, HALF_WIDTH)
    assert naive.pinned(cells, integral=False) == []
    assert naive.pinned(cells) == list(cells)
    assert [naive.value_range(cell)[0] for cell in cells] == [4, 0, 1, 0, 0, 0]

    spec = ReleaseSpec(
        (
            Table(("size",), "histogram", ordered=True),
            Continuous(("stats",), "moments of the histogram", histogram=("size",)),
        )
    )
    out, pinned = engine_pins({"size": value_counts(values), "stats": dist}, spec, 3)
    assert out == {"size": {"status": "blanked", "total": 5}, "stats": None}
    assert pinned == []


def test_sibship_moments_do_not_undo_the_merged_tail():
    """``n=102``, bins ``{1: 32, 2+: 70}`` and the mean and sd admit one tail, sizes ``(3, 3, 4)``."""
    dist = numeric_distribution(np.array(SIBSHIP_VALUES))
    mean, sd = round(dist.mean, 4), round(dist.std, 4)

    naive = System()
    cells = {f"c{v}": v for v in range(1, 13)}
    for cell in cells:
        naive.var(cell)
    naive.fix("c1", 32)
    tail = [cell for cell in cells if cell != "c1"]
    naive.add(dict.fromkeys(tail, 1), 70, 70)
    assert naive.pinned(tail) == [], "the merged histogram alone pins nothing"
    add_moments(naive, cells, 102, mean, sd, HALF_WIDTH)
    assert naive.pinned(tail) == tail
    assert [naive.value_range(cell)[0] for cell in ("c2", "c3", "c4", "c5")] == [67, 2, 1, 0]

    spec = ReleaseSpec(
        (
            Table(("size",), "histogram", ordered=True),
            Continuous(("children_per_pair",), "moments of the histogram", histogram=("size",)),
        )
    )
    out, pinned = engine_pins({"size": value_counts(SIBSHIP_VALUES), "children_per_pair": dist}, spec, 5)
    assert out == {
        "size": {"status": "merged", "total": 102, "bins": {"1": 32, "2+": 70}},
        "children_per_pair": None,
    }
    assert pinned == []


def test_odds_ratio_does_not_invert_a_blanked_2x2():
    """A 2x2 blanked to its margins has one free cell, which the odds ratio fixes."""
    a, b, c, d = 2, 40, 30, 50
    odds_ratio = round(a * d / (b * c), 4)

    def naive_system() -> System:
        system = System()
        for cell in "abcd":
            system.var(cell)
        for pair, margin in (("ab", a + b), ("cd", c + d), ("ac", a + c), ("bd", b + d)):
            system.add(dict.fromkeys(pair, 1), margin, margin)
        return system

    assert naive_system().pinned("abcd") == [], "the margins alone pin nothing"

    def recomputed(first: int) -> float:
        second, third = a + b - first, a + c - first
        fourth = c + d - third
        return first * fourth / (second * third) if second * third else float("inf")

    naive = naive_system()
    matches = close_to(odds_ratio, HALF_WIDTH)
    add_nonlinear(naive, ["a"], [(first,) for first in range(min(a + b, a + c) + 1)], lambda v: matches(recomputed(v)))
    assert naive.pinned("abcd") == ["a", "b", "c", "d"]
    assert naive.value_range("a") == (2, 2)

    spec = ReleaseSpec(
        (
            Contingency(("table",), "mate-pair 2x2"),
            Derived(("odds_ratio",), "computed from the 2x2", sources=(("table",),)),
        )
    )
    source = {"table": {"r1": {"c1": a, "c2": b}, "r2": {"c1": c, "c2": d}}, "odds_ratio": odds_ratio}
    out, pinned = engine_pins(source, spec, 5)
    assert out == {"table": {"status": "blanked", "total": 122}, "odds_ratio": None}
    assert pinned == []


def test_sex_ratio_ne_does_not_invert_a_blanked_sex_count():
    """``ne_sex_ratio = 4 Nm Nf / (Nm + Nf)`` with ``Nf`` released gives the blanked ``Nm``."""
    n_male, n_female, min_cell = 2, 40, 5
    ne = round(4 * n_male * n_female / (n_male + n_female), 4)

    def naive_system() -> System:
        system = System()
        system.var("m", 1, min_cell - 1)
        system.var("f")
        system.fix("f", n_female)
        return system

    assert naive_system().pinned(["m"]) == [], "a blanked count alone has min_cell - 1 values"

    naive = naive_system()
    matches = close_to(ne, HALF_WIDTH)
    add_nonlinear(
        naive, ["m"], [(m,) for m in range(1, min_cell)], lambda m: matches(4 * m * n_female / (m + n_female))
    )
    assert naive.pinned(["m"]) == ["m"]
    assert naive.value_range("m") == (2, 2)

    spec = ReleaseSpec(
        (
            Count(("n_male",), "sex count"),
            Count(("n_female",), "sex count"),
            Derived(("ne_sex_ratio",), "computed from both sex counts", sources=(("n_male",), ("n_female",))),
        )
    )
    out, pinned = engine_pins({"n_male": n_male, "n_female": n_female, "ne_sex_ratio": ne}, spec, min_cell)
    assert out == {"n_male": None, "n_female": 40, "ne_sex_ratio": None}
    assert pinned == []


# ---------------------------------------------------------------------------
# Property
# ---------------------------------------------------------------------------

HISTOGRAM_SPEC = ReleaseSpec((Table(("h",), "histogram", ordered=True),))


@given(
    counts=st.lists(st.integers(min_value=0, max_value=12), min_size=1, max_size=8),
    min_cell=st.integers(min_value=3, max_value=7),
)
def test_protect_ordered_histogram_properties(counts, min_cell):
    """No small bin is released, the total survives, protection is idempotent, and no merged-away cell is pinned."""
    table = ordered({str(i): count for i, count in enumerate(counts)})
    out = protect(table, min_cell)
    total = sum(counts)

    assert all(b.count == 0 or b.count >= min_cell for b in out.bins)
    assert out.total == (None if 0 < total < min_cell else total)
    if out.status is Status.BLANKED:
        assert out.bins == ()
        assert any(0 < count < min_cell for count in counts)
    else:
        assert sum(b.count for b in out.bins) == total
        assert [label for b in out.bins for label in b.covers] == [str(i) for i in range(len(counts))]
        assert (out.status is Status.INTACT) == all(len(b.covers) == 1 for b in out.bins)
        again = protect(ordered({b.label: b.count for b in out.bins}), min_cell)
        assert again.status is Status.INTACT
        assert [(b.label, b.count) for b in again.bins] == [(b.label, b.count) for b in out.bins]

    system, hidden = system_from_release(HISTOGRAM_SPEC, {("h",): out}, min_cell, cap=max(total, 1))
    assert (out.status is Status.INTACT) == (not hidden)
    assert system.pinned(hidden) == []


def test_blanked_count_is_never_pinned_at_the_floor():
    """At the smallest allowed minimum cell a blanked count still has two candidate values."""
    spec = ReleaseSpec((Count(("n",), "count"),))
    blanked = protect_count(2, MIN_CELL_FLOOR)
    assert blanked == ProtectedCount(Status.BLANKED, None)
    system, hidden = system_from_release(spec, {("n",): blanked}, MIN_CELL_FLOOR)
    assert hidden == [(("n",), None)]
    assert system.value_range(hidden[0]) == (1, 2)


# ---------------------------------------------------------------------------
# Review findings S3, S6, S7 and the taint between tables (S1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("held", ["missing", "none"])
def test_a_declared_support_the_payload_lacks_withholds(held):
    """A count or table whose rule names a support is never gated on its raw count when that support is not there."""
    source = payload()
    source["pairs"]["n_fs"] = 300
    if held == "missing":
        del source["pairs"]["n_fs_families"]
        del source["mates"]["families"]
    else:
        source["pairs"]["n_fs_families"] = None
        source["mates"]["families"] = None
    out = build_release(source, SPEC, 5)
    assert out["pairs"] == {"n_fs": None}
    assert out["mates"] == {"table": {"status": "blanked", "total": None}, "odds_ratio": None}

    spec = ReleaseSpec((Table(("by_code",), "pairs by code", ordered=False, support=("families",)),))
    assert build_release({"by_code": {"FS": 300, "MHS": 200}}, spec, 5) == {
        "by_code": {"status": "blanked", "total": None}
    }


def test_a_leaf_held_as_none_is_written_as_null():
    """S3: a statistic that was not computed is written as a withheld one is, so the key set is not data."""
    source = payload()
    source["mates"]["odds_ratio"] = None
    source["ne_sex_ratio"] = None
    source["inbreeding"]["F"] = None
    out = build_release(source, SPEC, 5)
    assert out["mates"]["odds_ratio"] is None
    assert out["ne_sex_ratio"] is None
    assert out["inbreeding"] == {"F": None}
    withheld = payload()
    withheld["sex"]["male"] = 2
    assert set(build_release(withheld, SPEC, 5)) == set(out)


@given(
    grid=st.integers(min_value=1, max_value=4).flatmap(
        lambda n_cols: st.lists(
            st.lists(st.integers(min_value=0, max_value=9), min_size=n_cols, max_size=n_cols), min_size=1, max_size=4
        )
    ),
    min_cell=st.integers(min_value=3, max_value=7),
)
def test_contingency_zero_rule_follows_from_the_small_cell_rule(grid, min_cell):
    """S7: a table is blanked exactly by a nonzero cell below the minimum, which covers every zero of a small margin."""
    rows = tuple(f"r{i}" for i in range(len(grid)))
    cols = tuple(f"c{j}" for j in range(len(grid[0])))
    out = protect(ContingencyTable(rows, cols, tuple(tuple(row) for row in grid)), min_cell)
    assert (out.status is Status.BLANKED) == any(0 < cell < min_cell for row in grid for cell in row)
    col_totals = [sum(col) for col in zip(*grid, strict=True)]
    zero_in_a_small_margin = any(
        cell == 0 and (0 < sum(row) < min_cell or 0 < col_totals[j] < min_cell)
        for row in grid
        for j, cell in enumerate(row)
    )
    if zero_in_a_small_margin:
        assert out.status is Status.BLANKED


REQUIRES_SPEC = ReleaseSpec(
    (
        Table(("by_status",), "margin of the cross table", ordered=False, requires=(("by_depth",),)),
        Contingency(("by_sex",), "another split of the margin", requires=(("by_depth",),)),
        Contingency(("by_depth",), "the cross table"),
    )
)


def test_a_table_follows_the_tables_it_requires():
    """A margin is released only while the table it is a margin of is intact; withheld, it keeps its total."""
    source = {
        "by_status": {"yes": 11, "no": 24},
        "by_sex": {"f": {"yes": 6, "no": 11}, "m": {"yes": 5, "no": 13}},
        "by_depth": {"0": {"yes": 10, "no": 10}, "1": {"yes": 1, "no": 9}, "2": {"yes": 0, "no": 5}},
    }
    out = build_release(source, REQUIRES_SPEC, 5)
    assert out == {
        "by_status": {"status": "blanked", "total": 35},
        "by_sex": {"status": "blanked", "total": 35},
        "by_depth": {"status": "blanked", "total": 35},
    }
    source["by_depth"]["1"] = {"yes": 5, "no": 5}
    source["by_status"] = {"yes": 15, "no": 20}
    out = build_release(source, REQUIRES_SPEC, 5)
    assert out["by_status"] == {"status": "intact", "total": 35, "bins": {"yes": 15, "no": 20}}
    assert out["by_sex"]["status"] == "intact"
    del source["by_depth"]
    assert build_release(source, REQUIRES_SPEC, 5)["by_status"] == {"status": "blanked", "total": 35}


def test_spec_rejects_bad_requirements():
    """A required table must be declared and must not itself wait on another."""
    margin = Table(("m",), "margin", ordered=False, requires=(("t",),))
    with pytest.raises(ValueError, match="table rule m refers to t"):
        ReleaseSpec((margin,))
    with pytest.raises(ValueError, match="requires t, which has requirements of its own"):
        ReleaseSpec((margin, Table(("t",), "table", ordered=False, requires=(("m",),))))


def test_a_link_with_no_parts_is_a_structural_zero():
    """With the deepest depth's cell declared zero, the column total and the founders' released row pin the middle row."""
    cells = Contingency(("by_depth",), "depth by status")
    status = Table(("by_status",), "status", ordered=False)
    source = {
        "by_status": {"yes": 11, "no": 24},
        "by_depth": {"0": {"yes": 10, "no": 10}, "1": {"yes": 1, "no": 9}, "2": {"yes": 0, "no": 5}},
    }
    column = Link(
        (("by_status",), "yes"), tuple((("by_depth",), (d, "yes")) for d in "012"), "the column is the margin"
    )
    deepest = Link((("by_depth",), ("2", "yes")), (), "the deepest depth has no offspring")
    middle = (("by_depth",), ("1", "yes"))

    def pinned(*links: Link) -> list:
        spec = ReleaseSpec((status, cells), links)
        system, hidden = system_from_release(spec, protect_payload(source, spec, 5), 5, cap=35)
        system.add({(("by_depth",), ("0", "yes")): 1}, 10, 10)
        return system.pinned([cell for cell in hidden if cell == middle])

    assert pinned(column) == []
    assert pinned(column, deepest) == [middle]


def test_f5_a_nonzero_cell_with_no_support_is_small_in_every_table_kind():
    """A count of 10 backed by no unit is withheld as a contingency cell, as it is as a bin and as a scalar."""
    grid = ContingencyTable(("a", "b"), ("x", "y"), ((10, 10), (10, 10)), support=((0, 9), (9, 9)))
    out = protect(grid, 5)
    assert (out.status, out.counts, out.total) == (Status.BLANKED, None, 40)
    flat = FreqTable((("ax", 10), ("ay", 10), ("bx", 10), ("by", 10)), ordered=False, support=(0, 9, 9, 9))
    assert protect(flat, 5).status is Status.BLANKED
    assert protect_count(10, 5, 0).value is None
