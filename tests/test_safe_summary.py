"""``summarize --safe-attempt``: the release list, the audit leaks it closes, and the exact oracle over its links.

``tests/data/safe_attempt/adversarial.tsv`` is ``example_pedigree.tsv`` with
the first two children of each sibship marked as a multiple-birth set, a
second component of three individuals, and one sibship of four born to two
deepest-depth individuals. An ordinary run on it shows every audit leak the
plan lists for ``summarize``; each leak test first asserts that, then that
the safe run does not.
"""

from __future__ import annotations

import copy
import csv
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import property_budget, run_pedsum, write_ped, written_output
from disclosure_oracle import add_mean, add_moments, system_from_release
from hypothesis import given
from test_size_structure_properties import acyclic_pedigree_frames

from pedsum import safe_summary
from pedsum.base import SEX_FEMALE, SEX_MALE
from pedsum.disclosure import (
    DEFAULT_MIN_CELL,
    Contingency,
    Count,
    ReleaseSpec,
    Table,
    build_release,
    protect_payload,
)
from pedsum.report import output_name
from pedsum.safe_summary import NOT_COMPUTED, build_source, safe_outputs, summarize_spec

ADVERSARIAL = Path(__file__).parent / "data" / "safe_attempt" / "adversarial.tsv"
FULL = [
    *("--multiple-birth-col", "set", "--birth-year-col", "birth_year"),
    *("--sex-concordance", "--per-individual-burden", "--tsv"),
]
OUTPUTS = ("summary.yaml", "summary.extra.yaml", "summary.pedigree.tsv", "summary.individual.tsv")

SIZE = ("pedigree", "structure", "size_structure")
DEPTH = (*SIZE, "depth")
#: Each released statistic that is a moment of a table's cells, with that table.
MOMENTS = {(*SIZE, "depth_stats"): DEPTH}
BY_DEPTH = ("pedigree", "strata", "depth_summary")
DEPTH_PARENTS = (*BY_DEPTH, "parents_recorded")
DEPTH_REPRODUCTIVE = (*BY_DEPTH, "reproductive")
DEPTH_INBRED = (*BY_DEPTH, "inbred")
#: The weight of each depth-by-recorded-parents cell in a count of parent links.
PARENT_LINKS = {"both": 2, "mother_only": 1, "father_only": 1}
#: Serialised precision: ``report._round_floats`` writes 4 decimals.
HALF_WIDTH = 0.5e-4

#: Integer leaves that are not counts of units.
NOT_COUNTS = {"max_degree_enumerated", "min_group_size", "requested", "completed", "seed", "q1", "median", "q3"}


def summarize(ped: Path, out_dir: Path, *flags: str, safe: bool = True, edit=None) -> dict[str, Any] | None:
    """Run ``summarize`` and return the safe source payload it built (None for an ordinary run).

    ``edit`` changes the source before it is released.
    """
    captured: list[dict[str, Any]] = []

    def capture(**kwargs):
        source = build_source(**kwargs)
        if edit is not None:
            edit(source)
        captured.append(source)
        return source

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(safe_summary, "build_source", capture)
        args = ["summarize", "--in", str(ped), "--out", str(out_dir), *flags, *(["--safe-attempt"] if safe else [])]
        res = run_pedsum(args)
    assert res.returncode == 0, res.stderr
    return captured[0] if safe else None


def load(out_dir: Path) -> dict[str, Any]:
    """The two summary YAMLs of a run, by file name."""
    return {name: yaml.safe_load(written_output(out_dir, name).read_text()) for name in OUTPUTS[:2]}


def text(out_dir: Path) -> str:
    """Every output file of a run, concatenated."""
    return "\n".join(path.read_text() for path in sorted(out_dir.iterdir()))


def leaves(obj: Any, path: tuple[str, ...] = ()):
    """Every ``(path, leaf)`` of a nested mapping."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, (*path, str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from leaves(value, (*path, str(i)))
    else:
        yield path, obj


def small_counts(out_dir: Path, min_cell: int) -> list[tuple[str, ...]]:
    """Paths of every integer in the outputs that is in ``1..min_cell-1`` and is not a known non-count."""
    found = []
    for name, payload in load(out_dir).items():
        found += [
            (name, *path)
            for path, leaf in leaves(payload)
            if isinstance(leaf, int)
            and not isinstance(leaf, bool)
            and 0 < leaf < min_cell
            and path[-1] not in NOT_COUNTS
        ]
    for name in OUTPUTS[2:]:
        with written_output(out_dir, name).open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                leaf = (row["subkey"] or row["key"]).split(".")[-1]
                if row["value"].isdecimal() and 0 < int(row["value"]) < min_cell and leaf not in NOT_COUNTS:
                    found.append((name, row["section"], row["key"], row["subkey"]))
    return found


def at(payload: Any, path: tuple[str, ...]) -> Any:
    """The value at ``path``, or None when a key is missing."""
    for key in path:
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload


def sensitive(cell, source: dict[str, Any], spec: ReleaseSpec, min_cell: int) -> bool:
    """Whether ``cell`` is one the minimum-cell rule protects: a small count, or a zero of a small row or column.

    A cell a link declares zero by construction is not: the pedigree's
    shape fixes it, whatever the data.
    """
    path, label = cell
    if any(link.whole == cell and not link.parts for link in spec.links):
        return False
    rule = next(rule for rule in spec.rules if rule.path == path)
    value = at(source, path)
    if isinstance(rule, Count):
        return True
    if isinstance(rule, Table):
        count = value[label]
        support = count if rule.support is None else at(source, rule.support)[label]
        return count > 0 and min(count, support) < min_cell
    assert isinstance(rule, Contingency)
    row, col = label
    if value[row][col]:
        return value[row][col] < min_cell
    totals = (sum(value[row].values()), sum(cells[col] for cells in value.values()))
    # A zero whose whole row or column is zero follows from that released zero total (D6: zeros of a 1-D table
    # stay visible), so it tells nothing about the small margin it also sits in.
    return 0 not in totals and any(total < min_cell for total in totals)


def pinned_sensitive(source: dict[str, Any], min_cell: int, spec: ReleaseSpec | None = None) -> list:
    """The protected cells a reader of the release can compute exactly.

    The reader's system is every released bin, total and declared link, plus
    the mean and sd of each released statistic of a table's cells. A hidden
    cell that is neither small nor a zero of a small margin (a large cell of
    a table withheld whole) may be pinned: the rule would have released it.
    """
    spec = spec or summarize_spec(source)
    protected = protect_payload(source, spec, min_cell)
    release = build_release(source, spec, min_cell)
    system, hidden = system_from_release(spec, protected, min_cell, cap=max(source["n_total"], 3) ** 2)
    for stats_path, table_path in MOMENTS.items():
        stats = at(release, stats_path)
        if stats and "mean" in stats:
            n = sum(at(source, table_path).values())
            cells = {(table_path, label): int(label) for label in protected[table_path].labels}
            add_moments(system, cells, n, round(stats["mean"], 4), round(stats["std"], 4), HALF_WIDTH)
    depths = at(source, DEPTH)
    # Depth 1's ancestors are its recorded parents; with two depths, the founders' descendant paths are those links.
    link_means = {"mean_distinct_ancestors": "1"} | ({"mean_descendant_paths": "0"} if len(depths) == 2 else {})
    for name, depth in link_means.items():
        means = at(release, (*BY_DEPTH, name))
        if means and "1" in depths:
            weights = {(DEPTH_PARENTS, ("1", col)): weight for col, weight in PARENT_LINKS.items()}
            add_mean(system, weights, depths[depth], round(means[depth], 4), HALF_WIDTH)
    return system.pinned([cell for cell in hidden if sensitive(cell, source, spec, min_cell)])


@pytest.fixture(scope="module")
def plain(tmp_path_factory) -> dict[str, Any]:
    """The ordinary summary YAMLs of the adversarial pedigree."""
    out_dir = tmp_path_factory.mktemp("plain")
    summarize(ADVERSARIAL, out_dir, *FULL, safe=False)
    return load(out_dir)


@pytest.fixture(scope="module")
def safe(tmp_path_factory) -> tuple[dict[str, Any], Path]:
    """The safe source payload and output directory of the adversarial pedigree."""
    out_dir = tmp_path_factory.mktemp("safe")
    source = summarize(ADVERSARIAL, out_dir, *FULL)
    assert source is not None
    return source, out_dir


# ---------------------------------------------------------------------------
# Release list
# ---------------------------------------------------------------------------


def test_safe_run_writes_the_four_reports_and_no_individual_table(safe):
    """The safe run writes the two reports and, under --tsv, the two long tables, and never annotated.tsv.gz."""
    _, out_dir = safe
    assert sorted(path.name for path in out_dir.iterdir()) == sorted(output_name(name, safe=True) for name in OUTPUTS)


def test_every_rule_has_a_reason(safe):
    """Each released leaf carries a sentence, not a placeholder."""
    source, _ = safe
    for rule in summarize_spec(source).rules:
        assert len(rule.reason.split()) >= 6, rule.path
        assert rule.reason.endswith("."), rule.path


def _cell_sum(source: dict[str, Any], ref) -> int:
    """The count a link's cell reference names: one bin, one contingency cell, or a table's total."""
    path, label = ref
    value = at(source, path)
    if label is None:
        return value if isinstance(value, int) else sum(leaf for _, leaf in leaves(value))
    return value[label[0]][label[1]] if isinstance(label, tuple) else value[label]


def test_links_are_true_and_cover_every_shared_margin(safe):
    """Each declared link holds in the data, and the links tie exactly the tables known to share cells.

    A link left out is a relation the oracle cannot see, so the set is pinned:
    every split of the individuals shares the total, each by-sex, by-depth
    and by-decade table has its margins, and the founders, the depth-0 row
    and the zero F bucket are tied to the cells that repeat them.
    """
    source, _ = safe
    spec = summarize_spec(source)
    for link in spec.links:
        assert link.reason.endswith("."), link
        if any(at(source, path) is None for path, _ in (link.whole, *link.parts)):
            continue  # a table this run did not produce, as the oracle skips it
        assert _cell_sum(source, link.whole) == sum(_cell_sum(source, part) for part in link.parts), link
    zeros = {link.whole for link in spec.links if not link.parts}
    deepest = list(at(source, DEPTH))[-1]
    assert zeros == {
        (DEPTH_REPRODUCTIVE, (deepest, "reproductive")),
        *((DEPTH_INBRED, (depth, "inbred")) for depth in "01"),
        *((DEPTH_PARENTS, (depth, "none")) for depth in at(source, DEPTH) if depth != "0"),
        *((DEPTH_PARENTS, ("0", col)) for col in PARENT_LINKS),
    }

    n, pedigree = ("n_total",), ("pedigree",)
    sex, parents, sex_source = (*SIZE, "sex"), (*SIZE, "parents_known"), (*SIZE, "sex_source")
    status = (*pedigree, "individuals", "reproduction", "by_status")
    inbred = (*pedigree, "relatedness", "inbreeding", "inbred")
    f_hist = (*pedigree, "relatedness", "inbreeding", "F_hist")
    births = (*pedigree, "demography", "multiple_births")
    by_sex, by_depth = (*pedigree, "strata", "sex_summary"), (*pedigree, "strata", "depth_summary")
    expected = {
        *((n, table) for table in (sex, parents, DEPTH, sex_source, status, inbred, f_hist)),
        (DEPTH, parents),
        (inbred, f_hist),
        ((*births, "twin_pairs_by_sex"), (*births, "twin_pairs_by_birth_decade")),
        *((margin, (*by_sex, name)) for name, margin in (("reproductive", status), ("inbred", inbred))),
        *((sex, (*by_sex, name)) for name in ("reproductive", "inbred")),
        *((margin, (*by_depth, name)) for name, margin in (("sex", sex), ("reproductive", status), ("inbred", inbred))),
        *((DEPTH, (*by_depth, name)) for name in ("sex", "reproductive", "inbred")),
        (parents, (*by_depth, "reproductive")),
        (parents, DEPTH_PARENTS),
        (DEPTH, DEPTH_PARENTS),
    }
    assert {(link.whole[0], link.parts[0][0]) for link in spec.links if link.parts} == expected

    depths, sexes = at(source, DEPTH), [label for label, count in at(source, sex).items() if count]
    margins = {(DEPTH, (*by_depth, "sex")): len(depths), (sex, (*by_depth, "sex")): len(sexes)}
    for key, n_labels in margins.items():
        assert sum((link.whole[0], link.parts[0][0]) == key for link in spec.links if link.parts) == n_labels, key


def test_a_key_no_rule_names_is_absent_from_every_output(tmp_path):
    """A key added beside any released leaf, at any level of the payload, reaches none of the four files."""
    marker = "INJECTED-7c1f"

    def inject(source: dict[str, Any]) -> None:
        rule_paths = {rule.path for rule in summarize_spec(source).rules}

        def walk(node: dict[str, Any], path: tuple[str, ...]) -> None:
            for key, value in list(node.items()):
                if isinstance(value, dict) and (*path, key) not in rule_paths:
                    walk(value, (*path, key))
            node[f"new_key_{len(path)}"] = marker
            node[f"new_section_{len(path)}"] = {"n": 1, "note": marker}

        walk(source, ())

    source = summarize(ADVERSARIAL, tmp_path, *FULL, edit=inject)
    assert source is not None
    assert marker in str(source["pedigree"]["relatedness"]["inbreeding"])
    assert marker not in text(tmp_path)
    assert "new_" not in text(tmp_path)


def test_no_released_count_is_below_the_minimum_cell(safe):
    """Walking every output generically finds no integer in 1..4 that counts units."""
    _, out_dir = safe
    assert small_counts(out_dir, DEFAULT_MIN_CELL) == []


def test_tables_hold_integer_counts(safe):
    """Every released bin and cell is an int: no share stands in for a count."""
    _, out_dir = safe
    bins = [
        leaf for payload in load(out_dir).values() for path, leaf in leaves(payload) if {"bins", "cells"} & set(path)
    ]
    assert bins
    assert all(isinstance(leaf, int) and not isinstance(leaf, bool) for leaf in bins)


def test_safe_build_is_idempotent(safe):
    """Releasing the same source twice gives the same outputs and leaves the source as it was."""
    source, _ = safe
    before = copy.deepcopy(source)
    first = safe_outputs(source, DEFAULT_MIN_CELL)
    assert safe_outputs(source, DEFAULT_MIN_CELL) == first
    assert source == before


def test_the_yamls_split_the_release_and_the_tsvs_repeat_it(safe):
    """Apart from the meta block, no leaf is in both YAMLs, and every TSV value is a YAML leaf."""
    source, _ = safe
    outputs = safe_outputs(source, DEFAULT_MIN_CELL)
    slim = {path for path, _ in leaves(outputs.slim) if len(path) > 1}
    extra = {path for path, _ in leaves(outputs.extra) if len(path) > 1}
    assert slim
    assert extra
    assert not slim & extra
    yaml_leaves = {(path[-1], leaf) for payload in (outputs.slim, outputs.extra) for path, leaf in leaves(payload)}
    tsv_leaves = {
        (path[-1], leaf) for payload in (outputs.pedigree_tsv, outputs.individual_tsv) for path, leaf in leaves(payload)
    }
    assert tsv_leaves == yaml_leaves


# ---------------------------------------------------------------------------
# Audit leaks (plan, "Audit findings this fixes"); leak 6 is effective-size
# ---------------------------------------------------------------------------


def _pedigree(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["summary.yaml"]["pedigree"]


def _extra(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["summary.extra.yaml"]["pedigree"]


def test_leak_1_no_share_implies_a_small_count(plain, safe):
    """Ordinary shares times their totals are counts of 1 to 4; the safe run releases neither histogram."""
    sibships = _pedigree(plain)["demography"]["sibship_size"]
    assert round(sibships["size_dist"]["4+"] * sibships["n_sibships"]) == 1
    females = _pedigree(plain)["strata"]["sex_summary"]["female"]["n"]
    share = _pedigree(plain)["individuals"]["reproduction"]["offspring_count_hist_female"]["3"]
    assert round(share * females) == 1

    _, out_dir = safe
    for key in ("size_dist", "sibship_size", "offspring_count_hist", "frac_", "_frac\t", "_frac:"):
        assert key not in text(out_dir), key


def test_leak_2_no_complement_recovers_a_withheld_count(plain, safe):
    """Ordinary depth rows keep n and n_reproductive beside a small n_terminal; the safe table is whole or absent."""
    rows = _extra(plain)["strata"]["depth_summary"]
    assert any(0 < row["n_terminal"] < DEFAULT_MIN_CELL for row in rows)
    inbreeding = _pedigree(plain)["relatedness"]["inbreeding"]
    n = plain["summary.yaml"]["n_total"]
    assert any(0 < round(share * n) < DEFAULT_MIN_CELL for share in inbreeding["hist"].values())

    source, out_dir = safe
    by_depth = _extra(load(out_dir))["strata"]["depth_summary"]["reproductive"]
    assert by_depth["status"] == "blanked"
    assert "cells" not in by_depth
    for key in ("n_terminal", "n_reproductive"):
        assert key not in text(out_dir), key
    f_hist = _extra(load(out_dir))["relatedness"]["inbreeding"]["F_hist"]
    assert f_hist == {"status": "blanked", "total": n}
    assert pinned_sensitive(source, DEFAULT_MIN_CELL) == []


def test_leak_3_no_maximum_f(plain, safe):
    """The ordinary run reports max_F overall and per depth; the safe run reports no extremum of F."""
    assert _pedigree(plain)["relatedness"]["inbreeding"]["max_F"] > 0
    assert any(row["max_F"] > 0 for row in _extra(plain)["strata"]["depth_summary"])

    _, out_dir = safe
    assert "max_F" not in text(out_dir)
    names = {path[-1] for payload in load(out_dir).values() for path, _ in leaves(payload)}
    assert not {name for name in names if name in ("min", "max") or name.startswith(("max_", "min_"))} - NOT_COUNTS


def test_leak_4_no_statistic_of_fewer_than_five_values(plain, safe):
    """The ordinary component_size summarises two components, exposing the small one; the safe run drops it."""
    component_size = _pedigree(plain)["structure"]["components"]["component_size"]
    assert component_size["nz"] == 2
    assert component_size["min"] == 3

    _, out_dir = safe
    for key in ("component_size", "largest_component", "next_components", "singletons"):
        assert key not in text(out_dir), key
    assert _pedigree(load(out_dir))["structure"]["components"] == {"n_components": None}


def test_leak_5_no_group_size_row_of_one_or_two_groups(plain, safe):
    """Ordinary by_group_size rows count 1 or 2 groups; the safe run has no per-size row."""
    concordance = _extra(plain)["demography"]["offspring_sex_concordance"]
    assert any(row["n_groups"] in (1, 2) for block in concordance.values() for row in block["by_group_size"])

    _, out_dir = safe
    for key in ("by_group_size", "group_size", "n_offspring", "n_within_group_pairs", "all_resolved"):
        assert key not in text(out_dir).replace("min_group_size", ""), key


def test_leak_7_no_largest_group_share(plain, safe):
    """The ordinary max_group_pair_share gives the largest group's size; the safe run has no such share."""
    sibship = _pedigree(plain)["demography"]["offspring_sex_concordance"]["sibship"]
    assert sibship["max_group_pair_share"] > 0

    _, out_dir = safe
    for key in ("max_group_pair_share", "largest_mate_network_share"):
        assert key not in text(out_dir), key


# ---------------------------------------------------------------------------
# D6, D9 and the sections the old per-section rules covered
# ---------------------------------------------------------------------------


def _couples(n_couples: int, children: int, start: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in range(n_couples):
        father, mother = start + k * (children + 2), start + k * (children + 2) + 1
        rows += [
            {"id": father, "sex": "M", "mother": -1, "father": -1},
            {"id": mother, "sex": "F", "mother": -1, "father": -1},
        ]
        rows += [
            {"id": mother + 1 + c, "sex": "MF"[(k + c) % 2], "mother": mother, "father": father}
            for c in range(children)
        ]
    return rows


def _pairs(release_dir: Path, source: dict[str, Any], min_cell: int) -> dict[str, Any]:
    return safe_outputs(source, min_cell).slim["pedigree"]["relatedness"]["relationship_pairs"]["by_code"]


def test_d9_one_sibship_of_three_withholds_its_pairs(tmp_path):
    """Three FS pairs from one sibship are one family: at minimum cell 3 the count passes and the family gate does not."""
    source = summarize(write_ped(tmp_path / "p.tsv", _couples(1, 3)), tmp_path / "out")
    assert source is not None
    pairs = source["pedigree"]["relatedness"]["relationship_pairs"]
    assert pairs["by_code"]["FS"] == 3
    assert pairs["support"]["FS"] == 1
    assert _pairs(tmp_path, source, 3) == {"status": "blanked", "total": None}


def test_d9_three_sibships_of_two_release_their_pairs(tmp_path):
    """Three FS pairs from three families pass at minimum cell 3."""
    source = summarize(write_ped(tmp_path / "p.tsv", _couples(3, 2)), tmp_path / "out")
    assert source is not None
    assert source["pedigree"]["relatedness"]["relationship_pairs"]["support"]["FS"] == 3
    released = _pairs(tmp_path, source, 3)
    assert released["status"] == "intact"
    assert released["bins"]["FS"] == 3
    assert "MO" not in released["bins"]
    assert "PO" not in released["bins"]


def test_d6_zero_bins_stay_in_a_histogram_and_blank_a_table_through_a_small_row(tmp_path):
    """A 1-D table shows its zero bins; a 2-D table with a zero in a row of two individuals is withheld whole."""
    rows = [*_couples(6, 2), *({"id": 900 + k, "sex": "", "mother": -1, "father": -1} for k in range(2))]
    source = summarize(write_ped(tmp_path / "p.tsv", rows), tmp_path / "out", "--allow-missing-sex")
    assert source is not None
    by_sex = source["pedigree"]["strata"]["sex_summary"]["reproductive"]
    assert by_sex["unknown"] == {"reproductive": 0, "terminal": 2}
    released = load(tmp_path / "out")["summary.yaml"]["pedigree"]
    assert released["strata"]["sex_summary"]["reproductive"] == {"status": "blanked", "total": 26}
    f_hist = load(tmp_path / "out")["summary.extra.yaml"]["pedigree"]["relatedness"]["inbreeding"]["F_hist"]
    assert f_hist == {
        "status": "intact",
        "total": 26,
        "bins": {"0": 26, "<0.0625": 0, "<0.125": 0, "<0.25": 0, "<1": 0},
    }


def test_concordance_does_not_tell_one_group_from_a_few(tmp_path):
    """One eligible group (not analysed) and two (analysed) release the same block: a withheld count and test."""
    blocks = []
    for n_couples in (1, 2):
        out_dir = tmp_path / f"out{n_couples}"
        source = summarize(
            write_ped(tmp_path / f"p{n_couples}.tsv", _couples(n_couples, 3)), out_dir, "--sex-concordance"
        )
        assert source is not None
        concordance = source["pedigree"]["demography"]["offspring_sex_concordance"]["sibship"]
        assert concordance["n_groups_eligible"] == n_couples
        assert (concordance["test"] == NOT_COMPUTED) == (n_couples == 1)
        blocks.append(load(out_dir)["summary.yaml"]["pedigree"]["demography"]["offspring_sex_concordance"]["sibship"])
    assert blocks[0] == blocks[1] == {"n_groups_eligible": None, "test": None}


def test_weinberg_follows_the_twin_pair_table(safe):
    """A small twin-pair cell withholds the table, Weinberg's estimate and the decade table with it."""
    source, out_dir = safe
    births = _pedigree(load(out_dir))["demography"]["multiple_births"]
    assert births["twin_pairs_by_sex"]["status"] == "intact"
    assert births["weinberg"]["mz_fraction_estimate"] is not None

    edited = copy.deepcopy(source)
    table = edited["pedigree"]["demography"]["multiple_births"]["twin_pairs_by_sex"]
    table["male_male"] += table["male_female"] - 3
    table["male_female"] = 3
    births = safe_outputs(edited, DEFAULT_MIN_CELL).slim["pedigree"]["demography"]["multiple_births"]
    assert births["twin_pairs_by_sex"] == {"status": "blanked", "total": 69}
    assert births["weinberg"] == {"mz_fraction_estimate": None}


def test_a_small_sex_source_class_withholds_the_table(safe):
    """Three individuals with a role-imputed sex are a small class: only the total is left."""
    source, _ = safe
    edited = copy.deepcopy(source)
    edited["pedigree"]["structure"]["size_structure"]["sex_source"] = {
        "input": 204,
        "imputed_from_missing": 0,
        "imputed_from_role": 3,
        "unresolved": 0,
    }
    released = safe_outputs(edited, DEFAULT_MIN_CELL).extra["pedigree"]["structure"]["size_structure"]
    assert released["sex_source"] == {"status": "blanked", "total": 207}


def test_pair_support_matches_the_reported_counts(safe):
    """Every reported code has a support, zero exactly when it has no pair and never above two per pair."""
    source, _ = safe
    pairs = source["pedigree"]["relatedness"]["relationship_pairs"]
    assert set(pairs["support"]) == set(pairs["by_code"])
    assert all(pairs["support"][code] <= 2 * count for code, count in pairs["by_code"].items())
    assert all((pairs["support"][code] == 0) == (count == 0) for code, count in pairs["by_code"].items())


# ---------------------------------------------------------------------------
# Exact oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_cell", [3, DEFAULT_MIN_CELL, 10])
def test_oracle_adversarial_pedigree(safe, min_cell):
    """No protected cell of the adversarial pedigree's release is pinned, at any minimum cell."""
    source, _ = safe
    assert pinned_sensitive(source, min_cell) == []


def _write_generated(frame, path: Path) -> Path:
    """Write a generated frame as a pedigree ``summarize`` accepts: no parent in both roles or of the wrong sex."""
    sex = frame["sex"].to_list()
    role: dict[int, int] = {}

    def parent(row: int, as_sex: int) -> int:
        """The parent's id, or -1 when it has the other sex or already has the other role."""
        other = SEX_MALE if as_sex == SEX_FEMALE else SEX_FEMALE
        if row == -1 or sex[row] == other or role.setdefault(row, as_sex) != as_sex:
            return -1
        return row + 1

    rows = [
        {
            "id": i + 1,
            "sex": {SEX_MALE: "M", SEX_FEMALE: "F"}.get(sex[i], ""),
            "mother": parent(mother, SEX_FEMALE),
            "father": parent(father, SEX_MALE),
        }
        for i, mother, father in frame.select("id", "mother", "father").iter_rows()
    ]
    return write_ped(path, rows)


@property_budget(40)
@given(frame=acyclic_pedigree_frames(max_size=70).filter(lambda frame: len(frame) > 0))
def test_oracle_generated_pedigrees(frame):
    """On generated pedigrees no protected cell is pinned, and no output holds a small count."""
    with tempfile.TemporaryDirectory() as tmp:
        ped = _write_generated(frame, Path(tmp) / "p.tsv")
        flags = ("--allow-missing-sex", "--sex-concordance", "--per-individual-burden", "--tsv")
        source = summarize(ped, Path(tmp) / "out", *flags)
        assert source is not None
        for min_cell in (3, DEFAULT_MIN_CELL):
            assert pinned_sensitive(source, min_cell) == []
        assert small_counts(Path(tmp) / "out", DEFAULT_MIN_CELL) == []


def test_oracle_catches_a_statistic_released_past_its_table(safe):
    """With depth's mean and sd released although its table was merged, the merged depths are pinned."""
    from pedsum.disclosure import Continuous

    source, _ = safe
    spec = summarize_spec(source)
    assert protect_payload(source, spec, DEFAULT_MIN_CELL)[DEPTH].status.value == "merged"
    ungated = tuple(
        Continuous(rule.path, rule.reason) if rule.path == (*SIZE, "depth_stats") else rule for rule in spec.rules
    )
    pinned = pinned_sensitive(source, DEFAULT_MIN_CELL, ReleaseSpec(ungated, spec.links))
    assert (DEPTH, "5") in pinned


def _graph(rows: list[dict[str, Any]]):
    """A graph over ``rows`` (ids 1..n in order) and its parent rows."""
    import numpy as np
    from pedigree_graph import PedigreeGraph

    mother_rows = np.array([row["mother"] - 1 if row["mother"] > 0 else -1 for row in rows], dtype=np.int64)
    father_rows = np.array([row["father"] - 1 if row["father"] > 0 else -1 for row in rows], dtype=np.int64)
    graph = PedigreeGraph.from_arrays(
        ids=np.arange(len(rows), dtype=np.int64),
        mother_ids=mother_rows,
        father_ids=father_rows,
        sex=np.array([SEX_MALE if row["sex"] == "M" else SEX_FEMALE for row in rows], dtype=np.int8),
    )
    return graph, mother_rows, father_rows


def _random_rows(seed: int, founders: int = 40, per_generation: int = 60, generations: int = 4) -> list[dict[str, Any]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = [{"id": i + 1, "sex": "FM"[i % 2], "mother": -1, "father": -1} for i in range(founders)]
    previous = rows
    for _ in range(generations):
        dams = [row["id"] for row in previous if row["sex"] == "F"]
        sires = [row["id"] for row in previous if row["sex"] == "M"]
        current = [
            {
                "id": len(rows) + k + 1,
                "sex": "FM"[int(rng.integers(2))],
                "mother": int(rng.choice(dams)),
                "father": int(rng.choice(sires)),
            }
            for k in range(per_generation)
        ]
        rows += current
        previous = current
    return rows


def _ancestors(row: int, up: int, mother_rows, father_rows) -> set[int]:
    """The rows exactly ``up`` generations above ``row``."""
    level = {row}
    for _ in range(up):
        level = {int(parent[r]) for r in level for parent in (mother_rows, father_rows) if parent[r] >= 0}
    return level


def test_pair_counts_agree_with_the_pair_lists():
    """One pass gives the counts of the pair lists, and a support no larger than the units those lists show.

    The units of a code, from its pair list: the families of its credited
    members, and the ancestors each pair goes through (the senior of a
    lineal pair, else the ancestors its two members share at the code's
    distances).
    """
    import numpy as np
    from pedigree_graph import RELATIONSHIPS

    graph, mother_rows, father_rows = _graph(_random_rows(7))
    family = safe_summary.family_ids(mother_rows, father_rows)
    pairs = safe_summary.pair_counts(graph, 5, mother_rows, father_rows)
    all_mothers = set(mother_rows.tolist())

    checked = 0
    for code, block in graph.relationship_pairs(max_degree=5).items():
        category = RELATIONSHIPS[code]
        assert pairs.counts[code] == len(block.first_rows), code
        if len(block.first_rows) == 0:
            assert pairs.support[code] == 0, code
            continue
        credited = block.first_rows if not category.symmetric else np.concatenate((block.first_rows, block.second_rows))
        families = int(np.unique(family[credited]).size)
        through: set[int] = set()
        for junior, senior in zip(block.first_rows.tolist(), block.second_rows.tolist(), strict=True):
            shared = _ancestors(junior, category.up, mother_rows, father_rows) & _ancestors(
                senior, category.down, mother_rows, father_rows
            )
            assert shared, code
            through |= shared
        mothers = through & all_mothers
        units = min(len(mothers), len(through - mothers)) if category.ancestor_count == 2 else len(through)
        assert 1 <= pairs.support[code] <= min(families, units), code
        checked += 1
    assert checked >= 15


def _person(i: int, sex: str, mother: int = -1, father: int = -1) -> dict[str, Any]:
    return {"id": i, "sex": sex, "mother": mother, "father": father}


def _one_person_shapes() -> dict[str, list[dict[str, Any]]]:
    """Pedigrees whose pairs of one code all hang on one person or one couple, across five families."""
    men, women = [_person(2 + k, "M") for k in range(5)], [_person(2 + k, "F") for k in range(5)]
    one_mother = [_person(1, "F"), *men, *(_person(7 + k, "F", 1, 2 + k) for k in range(5))]
    one_father = [_person(1, "M"), *women, *(_person(7 + k, "F", 2 + k, 1) for k in range(5))]
    # One couple (1, 2), five daughters (3..7), each with a husband (8..12) and one child (13..17).
    couple = [_person(1, "F"), _person(2, "M")]
    couple += [_person(3 + k, "F", 1, 2) for k in range(5)]
    couple += [_person(8 + k, "M") for k in range(5)]
    couple += [_person(13 + k, "F", 3 + k, 8 + k) for k in range(5)]
    # One woman with five husbands and two children by each.
    remarried = [_person(1, "F"), *men, *(_person(7 + 2 * k + j, "F", 1, 2 + k) for k in range(5) for j in range(2))]
    return {"MHS": one_mother, "PHS": one_father, "1C": couple, "Av": couple, "FS": remarried}


@pytest.mark.parametrize("code", ["MHS", "PHS", "1C", "Av", "FS"])
def test_s5_pairs_that_hang_on_one_person_have_a_support_of_one(code):
    """Five families' worth of pairs through one mother, one father or one couple are one unit, not five."""
    graph, mother_rows, father_rows = _graph(_one_person_shapes()[code])
    pairs = safe_summary.pair_counts(graph, 5, mother_rows, father_rows)
    assert pairs.counts[code] >= 5
    assert pairs.support[code] == 1


def test_s5_one_mother_of_five_half_sibs_releases_no_pair_count(tmp_path):
    """One woman with a child by each of five men gives ten MHS pairs from one mother, and they are withheld."""
    source = summarize(write_ped(tmp_path / "p.tsv", _one_person_shapes()["MHS"]), tmp_path / "out")
    assert source is not None
    pairs = source["pedigree"]["relatedness"]["relationship_pairs"]
    assert pairs["by_code"]["MHS"] == 10
    assert _pairs(tmp_path, source, 3) == {"status": "blanked", "total": None}


# ---------------------------------------------------------------------------
# Review findings S1 to S4: linked totals, per-depth means, the key set, the F moments
# ---------------------------------------------------------------------------


def _founders(n: int) -> list[dict[str, Any]]:
    """``n`` founders, ids 1..n, odd male and even female."""
    return [_person(i, "M" if i % 2 else "F") for i in range(1, n + 1)]


def _without_requirements(spec: ReleaseSpec) -> ReleaseSpec:
    """``spec`` with every table released on its own cells alone."""
    rules = tuple(replace(rule, requires=()) if getattr(rule, "requires", ()) else rule for rule in spec.rules)
    return ReleaseSpec(rules, spec.links)


def _with_sources(spec: ReleaseSpec, name: str, sources: tuple) -> ReleaseSpec:
    """``spec`` with the per-depth statistic ``name`` gated on ``sources`` only."""
    rules = tuple(replace(rule, sources=sources) if rule.path == (*BY_DEPTH, name) else rule for rule in spec.rules)
    return ReleaseSpec(rules, spec.links)


def test_s1_reproductive_total_is_not_released_beside_a_blanked_depth_table(tmp_path):
    """Ten of twenty founders and one depth-1 individual reproduce; the deepest depth cannot.

    The reproductive total (11) less the founders with descendants (10)
    would be the one reproductive individual at depth 1, a cell of the
    blanked depth-by-offspring table.
    """
    rows = _founders(20)
    rows += [_person(21 + k, "FM"[k % 2], 2 * (k // 2) + 2, 2 * (k // 2) + 1) for k in range(10)]
    rows += [_person(31 + k, "MF"[k % 2], 21) for k in range(5)]
    source = summarize(write_ped(tmp_path / "p.tsv", rows), tmp_path / "out")
    assert source is not None
    assert at(source, DEPTH_REPRODUCTIVE)["1"] == {"reproductive": 1, "terminal": 9}

    written = load(tmp_path / "out")
    strata = _extra(written)["strata"]["depth_summary"]
    assert strata["reproductive"] == {"status": "blanked", "total": 35}
    assert strata["mean_descendant_paths"] is None
    assert _pedigree(written)["structure"]["size_structure"]["parents_known"]["bins"]["founder_with_descendants"] == 10
    assert _pedigree(written)["individuals"]["reproduction"]["by_status"] == {"status": "blanked", "total": 35}
    assert _pedigree(written)["strata"]["sex_summary"]["reproductive"] == {"status": "blanked", "total": 35}

    middle = (DEPTH_REPRODUCTIVE, ("1", "reproductive"))
    assert pinned_sensitive(source, DEFAULT_MIN_CELL) == []
    assert middle in pinned_sensitive(source, DEFAULT_MIN_CELL, _without_requirements(summarize_spec(source)))


def test_s1_inbred_total_is_not_released_beside_a_blanked_depth_table(tmp_path):
    """Six of eight depth-2 individuals are inbred, and no one shallower can be.

    The inbred total (6) would leave the two who are not, a cell of the
    blanked depth-by-inbred table.
    """
    rows = _founders(12)
    # Full sibs 13 and 14 have six inbred children; the unrelated 15 and 16 have two who are not.
    rows += [_person(13, "M", 2, 1), _person(14, "F", 2, 1)]
    rows += [_person(15 + k, "MF"[k % 2], 2 * k + 4, 2 * k + 3) for k in range(4)]
    rows += [_person(19 + k, "MF"[k % 2], 14, 13) for k in range(6)]
    rows += [_person(25 + k, "MF"[k % 2], 16, 15) for k in range(2)]
    source = summarize(write_ped(tmp_path / "p.tsv", rows), tmp_path / "out")
    assert source is not None
    assert at(source, DEPTH_INBRED)["2"] == {"inbred": 6, "not_inbred": 2}

    written = load(tmp_path / "out")
    assert _extra(written)["strata"]["depth_summary"]["inbred"] == {"status": "blanked", "total": 26}
    inbreeding = _pedigree(written)["relatedness"]["inbreeding"]
    assert inbreeding == {"inbred": {"status": "blanked", "total": 26}, "mean_F_inbred": None}
    assert _extra(written)["relatedness"]["inbreeding"]["F_hist"] == {"status": "blanked", "total": 26}

    outbred = (DEPTH_INBRED, ("2", "not_inbred"))
    assert pinned_sensitive(source, DEFAULT_MIN_CELL) == []
    assert outbred in pinned_sensitive(source, DEFAULT_MIN_CELL, _without_requirements(summarize_spec(source)))


def _nine_with_both_parents_one_with_a_mother() -> list[dict[str, Any]]:
    rows = _founders(20)
    rows += [_person(21 + k, "F", 2 * (k % 5) + 2, 2 * (k % 5) + 1) for k in range(9)]
    return [*rows, _person(30, "F", 12)]


def test_s2_depth_means_follow_the_recorded_parents_table(tmp_path):
    """Nine of ten depth-1 individuals have both parents and one has a mother only.

    The mean ancestor count at depth 1 (1.9) times the depth's size (10)
    is 19 parent links, so one individual has one parent.
    """
    rows = _nine_with_both_parents_one_with_a_mother() + [_person(31 + k, "M", 21 + k) for k in range(5)]
    source = summarize(write_ped(tmp_path / "p.tsv", rows), tmp_path / "out")
    assert source is not None
    assert at(source, DEPTH_PARENTS)["1"] == {"both": 9, "mother_only": 1, "father_only": 0, "none": 0}
    assert at(source, (*BY_DEPTH, "mean_distinct_ancestors"))["1"] == pytest.approx(1.9)

    strata = _extra(load(tmp_path / "out"))["strata"]["depth_summary"]
    assert strata["parents_recorded"] == {"status": "blanked", "total": 35}
    assert strata["mean_distinct_ancestors"] is None
    assert strata["mean_descendant_paths"] is None

    one_parent = (DEPTH_PARENTS, ("1", "mother_only"))
    assert pinned_sensitive(source, DEFAULT_MIN_CELL) == []
    ungated = _with_sources(summarize_spec(source), "mean_distinct_ancestors", (DEPTH,))
    assert one_parent in pinned_sensitive(source, DEFAULT_MIN_CELL, ungated)


def test_s2_founders_descendant_paths_follow_the_recorded_parents_table(tmp_path):
    """With two depths the founders' mean descendant-path count is the same count of parent links."""
    source = summarize(write_ped(tmp_path / "p.tsv", _nine_with_both_parents_one_with_a_mother()), tmp_path / "out")
    assert source is not None
    assert at(source, (*BY_DEPTH, "mean_descendant_paths"))["0"] == pytest.approx(19 / 20)
    assert _extra(load(tmp_path / "out"))["strata"]["depth_summary"]["mean_descendant_paths"] is None

    assert pinned_sensitive(source, DEFAULT_MIN_CELL) == []
    ungated = _with_sources(summarize_spec(source), "mean_descendant_paths", (DEPTH,))
    leaked = at(build_release(source, ungated, DEFAULT_MIN_CELL), (*BY_DEPTH, "mean_descendant_paths"))
    depths = at(source, DEPTH)
    assert 2 * depths["1"] - round(leaked["0"] * depths["0"]) == 1


def _leaf_paths(out_dir: Path, spec: ReleaseSpec) -> set[tuple[str, ...]]:
    """The rule path of every leaf of a run's two reports: a table or a per-depth statistic is one leaf."""
    rules = {rule.path for rule in spec.rules}
    paths = set()
    for name, payload in load(out_dir).items():
        for path, _ in leaves(payload):
            cut = next((i for i in range(1, len(path) + 1) if path[:i] in rules), len(path))
            paths.add((name, *path[:cut]))
    return paths


def test_s3_the_key_set_does_not_depend_on_the_data(tmp_path):
    """No inbred individual (no mean), one (no sd) and three (both withheld) write the same keys."""
    key_sets = []
    for n_inbred in (0, 1, 3):
        rows = _founders(12)
        rows += [_person(13, "M", 2, 1), _person(14, "F", 2, 1)]
        rows += [_person(15 + k, "MF"[k % 2], 14, 13) for k in range(n_inbred)]
        rows += [_person(40 + k, "MF"[k % 2], 4, 3) for k in range(6)]
        out_dir = tmp_path / f"out{n_inbred}"
        source = summarize(write_ped(tmp_path / f"p{n_inbred}.tsv", rows), out_dir)
        assert source is not None
        inbreeding = source["pedigree"]["relatedness"]["inbreeding"]
        assert inbreeding["inbred"]["inbred"] == n_inbred
        assert (inbreeding["mean_F_inbred"] is None) == (n_inbred == 0)
        assert (inbreeding["sd_F_inbred"] is None) == (n_inbred < 2)
        key_sets.append(_leaf_paths(out_dir, summarize_spec(source)))
        assert ("summary.extra.yaml", "pedigree", "relatedness", "inbreeding", "sd_F_inbred") in key_sets[-1]
    assert key_sets[0] == key_sets[1] == key_sets[2]


def test_s3_weinberg_is_null_not_absent_without_twin_pairs(safe):
    """No set of two leaves the estimate undefined; it is written as a withheld one is."""
    source, _ = safe
    edited = copy.deepcopy(source)
    births = edited["pedigree"]["demography"]["multiple_births"]
    births["twin_pairs_by_sex"] = dict.fromkeys(births["twin_pairs_by_sex"], 0)
    births["weinberg"]["mz_fraction_estimate"] = None
    released = safe_outputs(edited, DEFAULT_MIN_CELL).slim["pedigree"]["demography"]["multiple_births"]
    assert released["weinberg"] == {"mz_fraction_estimate": None}


def test_s4_f_moments_are_not_released_beside_a_merged_f_histogram(safe):
    """Three at F = 0.25 and three at 0.0625 merge into one bucket; their mean (0.1562) would split it again."""
    source, _ = safe
    edited = copy.deepcopy(source)
    inbreeding = edited["pedigree"]["relatedness"]["inbreeding"]
    n = sum(inbreeding["inbred"].values())
    inbreeding["inbred"] = {"inbred": 6, "not_inbred": n - 6}
    inbreeding["F_hist"] = {"0": n - 6, "<0.0625": 3, "<0.125": 0, "<0.25": 3, "<1": 0}
    inbreeding["mean_F_inbred"], inbreeding["sd_F_inbred"] = 0.15625, 0.1027
    edited["pedigree"]["strata"]["depth_summary"]["inbred"] = {
        "0": {"inbred": 0, "not_inbred": n - 6},
        "2": {"inbred": 6, "not_inbred": 0},
    }
    outputs = safe_outputs(edited, DEFAULT_MIN_CELL)
    released = outputs.slim["pedigree"]["relatedness"]["inbreeding"]
    extra = outputs.extra["pedigree"]["relatedness"]["inbreeding"]
    assert released["inbred"]["bins"] == {"inbred": 6, "not_inbred": n - 6}
    assert extra["F_hist"]["status"] != "intact"
    assert released["mean_F_inbred"] is None
    assert extra["sd_F_inbred"] is None


@pytest.mark.parametrize("value", ["2", "0", "-1", "3.5", "x"])
def test_min_cell_below_three_is_a_usage_error(tmp_path, value):
    """``--min-cell`` refuses anything but an integer of at least 3, and writes nothing."""
    result = run_pedsum(
        ["summarize", "--in", str(ADVERSARIAL), "--out", str(tmp_path / "out"), "--safe-attempt", "--min-cell", value]
    )
    assert result.returncode == 2
    assert "--min-cell" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("min_cell", [3, 10])
def test_min_cell_sets_the_gate_and_is_recorded(tmp_path, min_cell):
    """The outputs are those of the requested minimum cell, hold no count below it, and say which it was."""
    source = summarize(ADVERSARIAL, tmp_path, *FULL, "--min-cell", str(min_cell))
    assert source is not None
    written = load(tmp_path)
    for name in ("summary.yaml", "summary.extra.yaml"):
        assert written[name]["safe_attempt_min_cell"] == min_cell
    expected = safe_outputs(source, min_cell)
    # Integer tables only: the writer rounds floats.
    for table in ("sex", "parents_known", "depth"):
        size = written["summary.yaml"]["pedigree"]["structure"]["size_structure"]
        assert size[table] == expected.slim["pedigree"]["structure"]["size_structure"][table]
    assert small_counts(tmp_path, min_cell) == []
    assert expected != safe_outputs(source, DEFAULT_MIN_CELL)


def test_safe_meta_names_paths_by_base_name_only(tmp_path):
    """The directories of the input and of every path on the command line reach no safe output."""
    ped = tmp_path / "project-x" / "ped.tsv"
    ped.parent.mkdir()
    ped.write_bytes(ADVERSARIAL.read_bytes())
    out = tmp_path / "researcher-y" / "out"
    for command, flags in (("summarize", ("--tsv",)), ("effective-size", ())):
        result = run_pedsum([command, "--in", str(ped), "--out", str(out / command), "--safe-attempt", *flags])
        assert result.returncode == 0, result.stderr
        written = text(out / command)
        assert "project-x" not in written
        assert "researcher-y" not in written
        assert str(tmp_path) not in written
        meta = yaml.safe_load(sorted((out / command).glob("*.txt"))[0].read_text())
        assert meta["input"] == "ped.tsv"
        assert (
            meta["command"] == f"pedsum {command} --in ped.tsv --out {command} --safe-attempt {' '.join(flags)}".strip()
        )
