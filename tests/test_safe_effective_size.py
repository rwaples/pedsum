"""``effective-size --safe-attempt``: the release list, audit leak 6, the support gates, and the exact oracle.

Two pedigrees carry most tests. ``tests/data/safe_attempt/adversarial.tsv``
(see ``test_safe_summary``) has a deepest depth of four full sibs, three of
them male, two individuals born 40 years before the rest, and one father
30 years older than his child: an ordinary run shows audit leak 6 on it.
``wide_rows`` is a random-mating pedigree in which every count an estimator
rests on is large, so a safe run releases all eight estimates.
"""

from __future__ import annotations

import copy
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest
import yaml
from conftest import property_budget, run_pedsum, write_ped, written_output
from disclosure_oracle import System, add_nonlinear, close_to, system_from_release
from hypothesis import given
from hypothesis import strategies as st
from pedigree_graph import PedigreeGraph
from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS
from test_safe_summary import _cell_sum, at, leaves, sensitive
from test_size_structure_properties import acyclic_pedigree_frames

from pedsum import safe_effective_size
from pedsum.base import SEX_FEMALE, SEX_MALE
from pedsum.cli import effective_size as cli
from pedsum.cli.parser import parse_args
from pedsum.disclosure import DEFAULT_MIN_CELL, Derived, build_release, protect_payload
from pedsum.memory import EXIT_MEMORY_LIMIT, GiB, MemoryWatchdog
from pedsum.safe_effective_size import WITHHELD, build_source, effective_size_spec, safe_payload
from pedsum.sections import compute_effective_size

if TYPE_CHECKING:
    from pedsum.disclosure import ReleaseSpec

ADVERSARIAL = Path(__file__).parent / "data" / "safe_attempt" / "adversarial.tsv"
BY = ("--birth-year-col", "birth_year")
ES = "effective_size"
DEPTH = ("supports", "depth")
SEX = ("supports", "sex")
DEPTH_SEX = ("supports", "depth_by_sex")
COHORT_SEX = ("supports", "birth_cohort_by_sex")
#: Serialised precision: ``report._round_floats`` writes 4 decimals.
HALF_WIDTH = 0.5e-4

#: Integer leaves that are not counts of units.
NOT_COUNTS = {"safe_attempt_min_cell"}


def wide_rows(seed: int = 7, founders: int = 12, per_generation: int = 80, generations: int = 4) -> list[dict]:
    """Founders born in one year, then generations of random matings born over three years each.

    ``ref`` marks every other individual of the last two generations.
    """
    rng = np.random.default_rng(seed)
    rows = [
        {"id": i + 1, "sex": "MF"[i % 2], "mother": -1, "father": -1, "birth_year": 1900, "ref": 0}
        for i in range(founders)
    ]
    previous = rows
    for g in range(1, generations + 1):
        dams = [row["id"] for row in previous if row["sex"] == "F"]
        sires = [row["id"] for row in previous if row["sex"] == "M"]
        current = [
            {
                "id": len(rows) + i + 1,
                "sex": "MF"[i % 2],
                "mother": int(rng.choice(dams)),
                "father": int(rng.choice(sires)),
                "birth_year": 1900 + 30 * g + i % 3,
                "ref": int(g >= generations - 1 and i % 2 == 0),
            }
            for i in range(per_generation)
        ]
        rows += current
        previous = current
    return rows


def run(ped: Path, out_dir: Path, *flags: str, safe: bool = True, edit=None, records=None) -> dict[str, Any] | None:
    """Run ``effective-size`` and return the safe source payload it built (None for an ordinary run).

    ``edit`` changes the source before it is released; ``records`` changes
    pedsum's estimator records before the source is built from them.
    """
    captured: list[dict[str, Any]] = []

    def capture(**kwargs):
        source = build_source(**kwargs)
        if edit is not None:
            edit(source)
        captured.append(source)
        return source

    def compute(pg, names):
        out = real_compute(pg, names)
        if records is not None:
            records(out)
        return out

    real_compute = cli.compute_effective_size
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(safe_effective_size, "build_source", capture)
        mp.setattr(cli, "compute_effective_size", compute)
        args = ["effective-size", "--in", str(ped), "--out", str(out_dir), *flags]
        res = run_pedsum([*args, *(["--safe-attempt"] if safe else [])])
    assert res.returncode == 0, res.stderr
    return captured[0] if safe else None


def load(out_dir: Path) -> dict[str, Any]:
    """The ``effective_size.yaml`` of a run."""
    return yaml.safe_load(written_output(out_dir, "effective_size.yaml").read_text())


def body(payload: dict[str, Any]) -> str:
    """The supports and estimates of a payload as text, without the meta block's paths and times."""
    return yaml.safe_dump({key: payload[key] for key in ("supports", ES) if key in payload})


def small_counts(payload: dict[str, Any], min_cell: int) -> list[tuple[str, ...]]:
    """Paths of every integer of the file that is in ``1..min_cell-1`` and is not a known non-count."""
    return [
        path
        for path, leaf in leaves(payload)
        if isinstance(leaf, int) and not isinstance(leaf, bool) and 0 < leaf < min_cell and path[-1] not in NOT_COUNTS
    ]


def released_tables(source: dict[str, Any], spec: ReleaseSpec, min_cell: int) -> dict:
    """The protected tables a reader of the file has: the gates are never written."""
    return {path: table for path, table in protect_payload(source, spec, min_cell).items() if path[0] != "gates"}


def sex_split_system(source: dict[str, Any], min_cell: int, spec: ReleaseSpec | None = None) -> tuple[System, list]:
    """What a reader of the file can write down, and the cells withheld from them.

    Every released bin, total and declared link, plus the one released
    estimate that is a closed form of table cells: the sex-ratio Ne of a
    depth, ``4 m f / (m + f)``.
    """
    spec = spec or effective_size_spec(source)
    release = build_release(source, spec, min_cell)
    cap = max(source["n_total"], 3) ** 2
    system, hidden = system_from_release(spec, released_tables(source, spec, min_cell), min_cell, cap=cap)
    sex_ratio = at(release, (ES, "ne_sex_ratio", "estimate"))
    if sex_ratio:
        for depth, ne in sex_ratio["ne_per_gen"].items():
            n = source["supports"]["depth"][depth]

            def consistent(m: int, f: int, ne=ne) -> bool:
                if ne is None:
                    return m == 0 or f == 0
                return m > 0 and f > 0 and close_to(round(ne, 4), HALF_WIDTH)(4 * m * f / (m + f))

            cells = [(DEPTH_SEX, (depth, "male")), (DEPTH_SEX, (depth, "female"))]
            add_nonlinear(system, cells, [(m, n - m) for m in range(n + 1)], consistent)
    founders = ("supports", "founders_contributing")
    if at(release, founders) is not None:
        # Not a sum of cells, so not a link: the contributing founders are among the founders, who are depth 0.
        system.add({(DEPTH, "0"): 1, (founders, None): -1}, 0, cap)
    return system, hidden


def pinned_sensitive(source: dict[str, Any], min_cell: int, spec: ReleaseSpec | None = None) -> list:
    """The protected cells a reader of the file can compute exactly; "protected" as ``test_safe_summary`` has it."""
    spec = spec or effective_size_spec(source)
    system, hidden = sex_split_system(source, min_cell, spec)
    return system.pinned([cell for cell in hidden if sensitive(cell, source, spec, min_cell)])


@pytest.fixture(scope="module")
def plain(tmp_path_factory) -> dict[str, Any]:
    """The ordinary ``effective_size.yaml`` of the adversarial pedigree."""
    out_dir = tmp_path_factory.mktemp("plain")
    run(ADVERSARIAL, out_dir, *BY, safe=False)
    return load(out_dir)


@pytest.fixture(scope="module")
def safe(tmp_path_factory) -> tuple[dict[str, Any], dict[str, Any]]:
    """The safe source payload and written file of the adversarial pedigree."""
    out_dir = tmp_path_factory.mktemp("safe")
    source = run(ADVERSARIAL, out_dir, *BY)
    assert source is not None
    assert [path.name for path in out_dir.iterdir()] == ["effective_size.txt"]
    return source, load(out_dir)


@pytest.fixture(scope="module")
def wide_ped(tmp_path_factory) -> Path:
    """The wide pedigree, written once."""
    return write_ped(tmp_path_factory.mktemp("wide") / "wide.tsv", wide_rows())


@pytest.fixture(scope="module")
def wide(wide_ped, tmp_path_factory) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The safe source, the safe file and the ordinary file of the wide pedigree."""
    out_dir = tmp_path_factory.mktemp("wide_out")
    source = run(wide_ped, out_dir / "safe", *BY)
    assert source is not None
    run(wide_ped, out_dir / "plain", *BY, safe=False)
    return source, load(out_dir / "safe"), load(out_dir / "plain")


# ---------------------------------------------------------------------------
# Release list
# ---------------------------------------------------------------------------


def test_every_rule_has_a_reason(safe):
    """Each released leaf carries a sentence, not a placeholder."""
    source, _ = safe
    for rule in effective_size_spec(source).rules:
        assert len(rule.reason.split()) >= 6, rule.path
        assert rule.reason.endswith("."), rule.path


def test_the_file_holds_the_release_and_no_gate(wide):
    """The file's top-level keys are fixed, every estimator has a record, and no gate is written."""
    source, written, _ = wide
    assert list(written) == [
        *("input", "command", "version", "generated_at", "n_total", "safe_attempt_min_cell"),
        *("status", "estimators_requested", "supports", ES),
    ]
    assert list(written[ES]) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
    assert source["gates"]
    for name in source["gates"]:
        assert name not in body(written), name
    assert "gates" not in written


def test_a_key_no_rule_names_is_absent_from_the_file(tmp_path):
    """A field pedigree-graph adds to a record, and a key added anywhere in the source, reach the file nowhere."""
    marker = "INJECTED-7c1f"

    def new_record_fields(records: dict[str, dict]) -> None:
        for record in records.values():
            record["new_field"] = marker
            record["new_table"] = {"ages_m": [75], "offspring_count_m": [1], "note": marker}

    def inject(source: dict[str, Any]) -> None:
        rules = {rule.path: rule for rule in effective_size_spec(source).rules}

        def walk(node: dict[str, Any], path: tuple[str, ...]) -> None:
            # Into each estimate's group of fields, but not into a field: a per-depth array is one value.
            if path not in rules:
                for key, value in list(node.items()):
                    rule = rules.get((*path, key))
                    if isinstance(value, dict) and (rule is None or isinstance(rule, Derived)):
                        walk(value, (*path, key))
            node[f"new_key_{len(path)}"] = marker
            node[f"new_section_{len(path)}"] = {"n": 1, "note": marker}

        walk(source, ())

    source = run(ADVERSARIAL, tmp_path, *BY, "--min-cell", "3", records=new_record_fields, edit=inject)
    assert source is not None
    assert "new_field" not in str({key: source[key] for key in source if not key.startswith("new_")})
    assert marker in str(source[ES]["ne_inbreeding"]["estimate"])
    assert marker in str(source["supports"])
    written = (tmp_path / "effective_size.txt").read_text()
    assert load(tmp_path)[ES]["ne_inbreeding"]["ne"] is not None
    assert marker not in written
    assert "new_" not in written


def test_links_are_true_and_cover_every_shared_margin(wide):
    """Each link holds in the data, and the links tie exactly the released tables that share cells."""
    source, _, _ = wide
    spec = effective_size_spec(source)
    for link in spec.links:
        assert link.reason.endswith("."), link
        assert _cell_sum(source, link.whole) == sum(_cell_sum(source, part) for part in link.parts), link
    n = ("n_total",)
    assert {(link.whole[0], link.parts[0][0]) for link in spec.links} == {
        *((n, table) for table in (DEPTH, SEX, COHORT_SEX)),
        (DEPTH, DEPTH_SEX),
        (SEX, DEPTH_SEX),
        (SEX, COHORT_SEX),
    }
    rows = {DEPTH_SEX: len(at(source, DEPTH)), COHORT_SEX: len(at(source, COHORT_SEX))}
    assert sum((link.whole[0], link.parts[0][0]) == (DEPTH, DEPTH_SEX) for link in spec.links) == rows[DEPTH_SEX]
    for table, n_rows in rows.items():
        columns = [link for link in spec.links if (link.whole[0], link.parts[0][0]) == (SEX, table)]
        assert [link.whole[1] for link in columns] == ["male", "female"]
        assert all(len(link.parts) == n_rows for link in columns)


def test_reference_table_shares_the_total(wide_ped, tmp_path):
    """Under --reference-col the used and unused individuals are a split of the pedigree, linked to its total."""
    source = run(wide_ped, tmp_path, "--estimators", "ne_individual_delta_f", "--reference-col", "ref")
    assert source is not None
    spec = effective_size_spec(source)
    assert {(link.whole[0], link.parts[0][0]) for link in spec.links} == {
        (("n_total",), DEPTH),
        (("n_total",), ("supports", "reference")),
    }
    assert sum(source["supports"]["reference"].values()) == source["n_total"]


def test_no_released_count_is_below_the_minimum_cell(safe, wide):
    """Walking each file generically finds no integer in 1..4 that counts units."""
    assert small_counts(safe[1], DEFAULT_MIN_CELL) == []
    assert small_counts(wide[1], DEFAULT_MIN_CELL) == []


def test_safe_build_is_idempotent(safe):
    """Releasing the same source twice gives the same file and leaves the source as it was."""
    source, _ = safe
    before = copy.deepcopy(source)
    first = safe_payload(source, DEFAULT_MIN_CELL)
    assert safe_payload(source, DEFAULT_MIN_CELL) == first
    assert source == before


def test_ordinary_run_never_loads_the_safe_module(tmp_path):
    """An ordinary run does not build a safe source: the release list is imported only under --safe-attempt."""
    assert run(ADVERSARIAL, tmp_path, *BY, safe=False) is None
    assert "safe_attempt_min_cell" not in load(tmp_path)


# ---------------------------------------------------------------------------
# Audit leak 6 and D7
# ---------------------------------------------------------------------------


def test_leak_6_no_parental_age_table_and_no_birth_year_window(plain, safe):
    """The ordinary run lists a single old father and the earliest birth year; the safe run has neither."""
    hill = plain[ES]["ne_hill_overlapping"]
    ages = hill["age_table"]
    assert ages["ages_m"][-1] == 30
    assert ages["offspring_count_m"][-1] == 1
    birth_years = pl.read_csv(ADVERSARIAL, separator="\t")["birth_year"]
    assert hill["cohort_window"]["c_min"] == birth_years.min() == 1960

    source, written = safe
    assert source[ES]["ne_hill_overlapping"]["estimate"]["ne"] is not None
    assert written[ES]["ne_hill_overlapping"]["ne"] is not None, "the estimate itself is released"
    text = body(written)
    for key in (
        *("age_table", "ages_m", "ages_f", "offspring_count", "cohort_window", "c_min", "c_max"),
        *("reproductive_age_p95", "n_excluded", "n_unknown_birth_year", "n_offspring_pairs"),
    ):
        assert key not in text, key
    assert str(birth_years.min()) not in text
    assert str(birth_years.max()) not in text
    cohorts = written["supports"]["birth_cohort_by_sex"]["cells"]
    assert all(count >= DEFAULT_MIN_CELL for cells in cohorts.values() for count in cells.values())


def test_d7_no_extremum_and_no_offspring_count_moment(plain, wide):
    """With every estimate released, the file still has no extremum and no moment of per-individual counts."""
    _, written, ordinary = wide
    assert all(record["ne"] is not None and "reason" not in record for record in written[ES].values())
    names = {path[-1] for path, _ in leaves(written)} | {key for path, _ in leaves(written) for key in path}
    dropped = {
        *("max_delta_final", "asymptote_reached", "census_ratio", "cohort_window", "age_table", "reference_generation"),
        *("v_mm", "v_mf", "v_fm", "v_ff", "cov_m", "cov_f", "Vk_m", "Vk_f", "kbar_m", "kbar_f"),
        *("Vk_m_per_cohort", "Vk_f_per_cohort", "N1_m", "N1_f", "N1_m_per_cohort", "N1_f_per_cohort"),
        *("n_eligible_cohorts", "slope", "n_depths_used", "n_cohorts", "final_depth", "mean_eqg_per_gen"),
        *("n_effective_founders", "sum_c_squared", "generation_interval", "n_reference", "n_used_per_gen"),
        *("n_male_per_gen", "n_female_per_gen", "n_genomes_per_gen", "depths", "parent_depths", "cohort_years"),
        *("transition_from", "transition_to", "no_estimate_code"),
    }
    assert dropped <= {key for record in ordinary[ES].values() for key in record} | set(
        plain[ES]["ne_hill_overlapping"]
    )
    assert not dropped & names
    assert not {name for name in names if name in ("min", "max") or name.startswith(("max_", "min_"))}


def test_released_estimates_are_the_ordinary_ones(wide):
    """A released value is the ordinary run's value, scalars and arrays alike."""
    _, written, ordinary = wide
    for name, record in written[ES].items():
        theirs = ordinary[ES][name]
        assert record["ne"] == theirs["ne"], name
    assert (
        list(written[ES]["ne_inbreeding"]["mean_f_per_gen"].values()) == ordinary[ES]["ne_inbreeding"]["mean_f_per_gen"]
    )
    assert list(written[ES]["ne_coancestry"]["ne_per_gen"]) == ["0-1", "1-2", "2-3", "3-4"]
    hill, theirs = written[ES]["ne_hill_overlapping"], ordinary[ES]["ne_hill_overlapping"]
    assert list(hill["ne_per_cohort"].values()) == theirs["ne_per_cohort"]
    assert [int(year) for year in hill["ne_per_cohort"]] == theirs["cohort_years"]
    assert (hill["T_m"], hill["T_f"]) == (theirs["T_m"], theirs["T_f"])


# ---------------------------------------------------------------------------
# Supports and gates
# ---------------------------------------------------------------------------


def test_supports_are_the_counts_the_estimators_report(wide):
    """Each support and gate pedigree-graph also reports is the number it reports."""
    source, _, ordinary = wide
    es = ordinary[ES]
    supports, gates = source["supports"], source["gates"]
    sex_ratio = es["ne_sex_ratio"]
    assert [cells["male"] for cells in supports["depth_by_sex"].values()] == sex_ratio["n_male_per_gen"]
    assert [cells["female"] for cells in supports["depth_by_sex"].values()] == sex_ratio["n_female_per_gen"]
    assert list(supports["depth"]) == [str(depth) for depth in sex_ratio["depths"]]
    assert list(gates["genomes_by_depth"].values()) == es["ne_group_coancestry"]["n_genomes_per_gen"]
    assert list(gates["delta_f_used_by_depth"].values()) == es["ne_individual_delta_f"]["n_used_per_gen"]
    assert gates["reference_used"] == es["ne_individual_delta_f"]["n_reference"]
    hill = es["ne_hill_overlapping"]
    assert sum(supports["parent_links_with_birth_years"].values()) == hill["n_offspring_pairs"]
    cohorts = dict(supports["birth_cohort_by_sex"])
    rest = cohorts.pop("not_used")
    assert [int(year) for year in cohorts] == hill["cohort_years"]
    assert [cells["male"] for cells in cohorts.values()] == hill["N1_m_per_cohort"]
    assert [cells["female"] for cells in cohorts.values()] == hill["N1_f_per_cohort"]
    assert rest["male"] + rest["female"] == source["n_total"] - sum(hill["N1_m_per_cohort"] + hill["N1_f_per_cohort"])
    assert "reference" not in supports, "the default reference is the deepest depth, already a cell of the depth table"


def _graph(rows: list[dict], twins: dict[int, int] | None = None) -> PedigreeGraph:
    ids = np.array([row["id"] for row in rows], dtype=np.int64)
    twin_ids = np.full(len(rows), -1, dtype=np.int64)
    for a, b in (twins or {}).items():
        twin_ids[a - 1], twin_ids[b - 1] = b, a
    return PedigreeGraph.from_arrays(
        ids=ids,
        mother_ids=np.array([row["mother"] for row in rows], dtype=np.int64),
        father_ids=np.array([row["father"] for row in rows], dtype=np.int64),
        sex=np.array([SEX_MALE if row["sex"] == "M" else SEX_FEMALE for row in rows], dtype=np.int8),
        birth_year=np.array([row["birth_year"] for row in rows], dtype=np.int32),
        twin_ids=twin_ids,
    )


def _mz_pairs(rows: list[dict], n_pairs: int) -> dict[int, int]:
    """The first ``n_pairs`` pairs of same-sex full sibs, each individual in at most one pair."""
    by_family: dict[tuple, int] = {}
    pairs: dict[int, int] = {}
    for row in rows:
        if row["mother"] == -1:
            continue
        key = (row["mother"], row["father"], row["sex"])
        if key in by_family and len(pairs) < n_pairs:
            pairs[by_family.pop(key)] = row["id"]
        else:
            by_family.setdefault(key, row["id"])
    assert len(pairs) == n_pairs
    return pairs


def test_gates_agree_with_a_brute_force_count_with_mz_twins():
    """On a pedigree with MZ co-twins, each count pedsum computes itself matches a direct, slower count."""
    rows = wide_rows(seed=3, founders=10, per_generation=30, generations=3)
    graph = _graph(rows, _mz_pairs(rows, 6))
    records = compute_effective_size(graph, ALL_EFFECTIVE_SIZE_ESTIMATORS)
    supports, gates = safe_effective_size._counts(graph, list(ALL_EFFECTIVE_SIZE_ESTIMATORS), records, None)

    assert list(gates["genomes_by_depth"].values()) == records["ne_group_coancestry"]["n_genomes_per_gen"]
    assert list(gates["genomes_by_depth"].values()) != list(supports["depth"].values())

    depth = np.asarray(graph.depth)
    mother, father = np.asarray(graph.mother_rows), np.asarray(graph.father_rows)
    n = len(rows)
    ancestors = set(np.flatnonzero(depth == depth.max()).tolist())
    frontier = set(ancestors)
    while frontier:
        frontier = {int(p) for row in frontier for p in (mother[row], father[row]) if p >= 0} - ancestors
        ancestors |= frontier
    contributing = sum(mother[row] < 0 and father[row] < 0 for row in ancestors)
    assert supports["founders_contributing"] == gates["founder_genomes_contributing"] == contributing
    rest = supports["depth"]["0"] - contributing
    assert gates["founders_floor"] == min(contributing, rest or contributing, *supports["depth"].values())

    fathers = {int(row) for row in father if row >= 0}
    by_depth = [sum(depth[row] == d for row in fathers) for d in range(int(depth.max()) + 1)]
    assert list(gates["fathers_by_depth"].values()) == by_depth

    twin = np.asarray(graph.twin_rows)
    theta = records["ne_coancestry"]["mean_theta_per_gen"]
    for d, bound in enumerate(gates["related_families_by_depth"].values()):
        genomes = [row for row in range(n) if depth[row] == d and not 0 <= twin[row] < row]
        sib_pairs = [
            (a, parent[a])
            for parent in (mother, father)
            for a in genomes
            for b in genomes
            if a != b and parent[a] == parent[b] >= 0
        ]
        families = {(mother[a], father[a]) for a, _ in sib_pairs}
        shared_parents = {parent for _, parent in sib_pairs}
        assert bound == (max(min(len(families), len(shared_parents)), 1) if theta[d] else 0), d
    assert gates["related_families_by_depth"]["0"] == 0
    assert gates["related_families_by_depth"]["1"] >= DEFAULT_MIN_CELL
    assert list(gates["cotwins_by_depth"].values()) == [int((depth[twin >= 0] == d).sum()) // 2 for d in range(4)]


@pytest.mark.parametrize(
    ("gate", "withheld", "kept"),
    [
        (("gates", "fathers_by_depth"), ["ne_variance_family_size"], ["ne_sex_ratio", "ne_hill_overlapping"]),
        (("gates", "mothers_by_depth"), ["ne_variance_family_size"], ["ne_sex_ratio"]),
        (
            ("gates", "inbred_by_depth"),
            ["ne_inbreeding", "ne_group_coancestry"],
            ["ne_coancestry", "ne_individual_delta_f"],
        ),
        (("gates", "related_families_by_depth"), ["ne_coancestry", "ne_group_coancestry"], ["ne_inbreeding"]),
        (("gates", "genomes_by_depth"), ["ne_coancestry", "ne_group_coancestry"], ["ne_inbreeding"]),
        (("gates", "fathers_by_birth_cohort"), ["ne_hill_overlapping"], ["ne_variance_family_size"]),
        (("gates", "mothers_by_birth_cohort"), ["ne_hill_overlapping"], ["ne_variance_family_size"]),
    ],
)
def test_one_small_gate_cell_withholds_each_estimate_that_rests_on_it(wide, gate, withheld, kept):
    """A count of 2 at one depth or cohort withholds the whole estimate, arrays and scalar, and no other estimator."""
    source, _, _ = wide
    edited = copy.deepcopy(source)
    table = at(edited, gate)
    table[list(table)[1]] = 2
    records = safe_payload(edited, DEFAULT_MIN_CELL)[ES]
    for name in withheld:
        assert records[name]["reason"] == WITHHELD, name
        assert records[name]["ne"] is None, name
        arrays = [value for key, value in records[name].items() if key.endswith(("_per_gen", "_per_cohort"))]
        assert all(value is None for value in arrays), name
    for name in kept:
        assert records[name]["ne"] is not None, name


@pytest.mark.parametrize(
    ("table", "withheld", "kept"),
    [
        (COHORT_SEX, ["ne_hill_overlapping"], ["ne_sex_ratio", "ne_variance_family_size"]),
        (DEPTH_SEX, ["ne_sex_ratio", "ne_variance_family_size"], ["ne_hill_overlapping", "ne_inbreeding"]),
    ],
)
def test_one_small_cell_of_a_by_sex_table_withholds_the_estimates_over_it(wide, table, withheld, kept):
    """Four males in one cohort or depth withhold the table and every estimate computed from it, and no other."""
    source, _, _ = wide
    edited = copy.deepcopy(source)
    cells = at(edited, table)
    cells[list(cells)[1]]["male"] = 4
    payload = safe_payload(edited, DEFAULT_MIN_CELL)
    assert payload["supports"][table[-1]]["status"] == "blanked"
    for name in withheld:
        assert payload[ES][name]["reason"] == WITHHELD, name
        assert all(value is None for key, value in payload[ES][name].items() if key.startswith(("ne", "Ne"))), name
    for name in kept:
        assert payload[ES][name]["ne"] is not None, name


def test_delta_f_parts_follow_their_own_counts(wide):
    """The per-depth Ne follows the per-depth counts, the scalar the reference, the lagged Ne the lagged count."""
    source, _, _ = wide

    def record(path: tuple[str, ...], value: Any) -> dict[str, Any]:
        edited = copy.deepcopy(source)
        node = edited
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return safe_payload(edited, DEFAULT_MIN_CELL)[ES]["ne_individual_delta_f"]

    per_depth = record(("gates", "inbred_by_depth", "2"), 4)
    assert per_depth["ne_per_gen"] is None
    assert per_depth["ne"] is not None
    assert "reason" not in per_depth

    for gate in ("reference_used", "reference_inbred"):
        scalar = record(("gates", gate), 4)
        assert (scalar["ne"], scalar["standard_error"], scalar["ne_unrelated_founders"]) == (None, None, None), gate
        assert scalar["reason"] == WITHHELD
        assert scalar["ne_per_gen"] is not None

    lagged = record(("gates", "reference_past_first_generation"), 4)
    assert lagged["ne_unrelated_founders"] is None
    assert lagged["ne"] is not None


@pytest.mark.parametrize(("contributing", "smaller"), [(4, 4), (10, 2)])
def test_few_founders_on_either_side_withhold_the_contribution_ne(wide, contributing, smaller):
    """Four contributing founders, or two that do not contribute, withhold the count and the estimate over it."""
    source, written, _ = wide
    assert written["supports"]["founders_contributing"] == source["gates"]["founders_floor"] == 12
    assert written[ES]["ne_long_term_contributions"]["ne"] is not None
    edited = copy.deepcopy(source)
    edited["supports"]["founders_contributing"] = contributing
    edited["gates"]["founders_floor"] = smaller
    payload = safe_payload(edited, DEFAULT_MIN_CELL)
    assert payload["supports"]["founders_contributing"] is None
    assert payload[ES]["ne_long_term_contributions"] == {"ne": None, "reason": WITHHELD}

    edited = copy.deepcopy(source)
    edited["gates"]["founder_genomes_contributing"] = 4
    assert safe_payload(edited, DEFAULT_MIN_CELL)[ES]["ne_long_term_contributions"]["reason"] == WITHHELD


def test_founder_count_does_not_give_back_a_withheld_depth(tmp_path):
    """Three founders and two children at minimum cell 3: the depth table is withheld, and the founders with it.

    The generated-pedigree oracle found this: a released count of founders
    is the size of depth 0, and the total less that is the other depth.
    """
    rows = [{"id": i + 1, "sex": "MFM"[i], "mother": -1, "father": -1} for i in range(3)]
    rows += [{"id": 4, "sex": "F", "mother": 2, "father": 1}, {"id": 5, "sex": "M", "mother": 2, "father": 3}]
    source = run(write_ped(tmp_path / "p.tsv", rows), tmp_path / "out", "--min-cell", "3")
    assert source is not None
    assert source["supports"]["depth"] == {"0": 3, "1": 2}
    assert source["supports"]["founders_contributing"] == 3
    written = load(tmp_path / "out")
    assert written["supports"]["depth"] == {"status": "blanked", "total": 5}
    assert written["supports"]["founders_contributing"] is None
    assert pinned_sensitive(source, 3) == []


def test_founders_that_do_not_contribute_set_the_floor(tmp_path):
    """Founders with no descendant in the deepest depth are the other side of the contributing count."""
    rows = [
        *wide_rows(),
        *(
            {"id": 900 + i, "sex": "MF"[i % 2], "mother": -1, "father": -1, "birth_year": 1900, "ref": 0}
            for i in range(2)
        ),
    ]
    source = run(write_ped(tmp_path / "p.tsv", rows), tmp_path / "out", "--estimators", "ne_long_term_contributions")
    assert source is not None
    assert source["supports"]["depth"]["0"] == 14
    assert source["supports"]["founders_contributing"] == 12
    assert source["gates"]["founders_floor"] == 2
    written = load(tmp_path / "out")
    assert written["supports"]["depth"]["bins"]["0"] == 14
    assert written["supports"]["founders_contributing"] is None
    assert written[ES]["ne_long_term_contributions"] == {"ne": None, "reason": WITHHELD}


def _one_father_rows(child_sexes: str) -> list[dict]:
    """One father born 1900, a mother born 1950 for each child, and the children, born 1980: paternal half sibs."""
    n = len(child_sexes)
    rows = [{"id": 1, "sex": "M", "mother": -1, "father": -1, "birth_year": 1900}]
    rows += [{"id": 2 + i, "sex": "F", "mother": -1, "father": -1, "birth_year": 1950} for i in range(n)]
    rows += [
        {"id": 2 + n + i, "sex": sex, "mother": 2 + i, "father": 1, "birth_year": 1980}
        for i, sex in enumerate(child_sexes)
    ]
    return rows


@pytest.mark.parametrize("child_sexes", ["MFMFM", "MMMMM", "MMMMMFFFFF"])
def test_e1_links_from_one_parent_release_no_generation_interval(tmp_path, child_sexes):
    """Review finding E1: five father links that are one father's gave ``T_m: 80.0``, his age at those births.

    The sex tables fail for the first two pedigrees and pass for the third
    (five of each sex at depth 1, a sixth male at depth 0 is the father:
    still small), so the interval is withheld on the parents alone.
    """
    source = run(write_ped(tmp_path / "p.tsv", _one_father_rows(child_sexes)), tmp_path / "out", *BY)
    assert source is not None
    n = len(child_sexes)
    assert source["supports"]["parent_links_with_birth_years"] == {"father": n, "mother": n}
    assert source["gates"]["parents_with_birth_years"] == {"father": 1, "mother": n}
    assert source[ES]["ne_hill_overlapping"]["interval"] == {"T_m": 80.0, "T_f": 30.0}

    written = load(tmp_path / "out")
    hill = written[ES]["ne_hill_overlapping"]
    assert hill["T_m"] is None
    assert hill["T_f"] is None
    assert hill["reason"] == WITHHELD
    assert written["supports"]["parent_links_with_birth_years"] == {"status": "blanked", "total": 2 * n}
    text = body(written)
    assert "80.0" not in text
    assert "30.0" not in text


def test_e1_interval_is_released_over_enough_distinct_parents(wide):
    """The parents behind the links are counted once each, and a pedigree with enough of them keeps its interval."""
    source, written, plain = wide
    parents = source["gates"]["parents_with_birth_years"]
    links = source["supports"]["parent_links_with_birth_years"]
    assert all(DEFAULT_MIN_CELL <= parents[role] < links[role] for role in ("father", "mother"))
    hill = written[ES]["ne_hill_overlapping"]
    assert hill["T_m"] == pytest.approx(plain[ES]["ne_hill_overlapping"]["T_m"], abs=HALF_WIDTH)

    edited = copy.deepcopy(source)
    edited["gates"]["parents_with_birth_years"]["father"] = DEFAULT_MIN_CELL - 1
    hill = safe_payload(edited, DEFAULT_MIN_CELL)[ES]["ne_hill_overlapping"]
    assert hill["T_m"] is None
    assert hill["ne"] is None
    assert hill["reason"] == WITHHELD


def test_e2_kinship_among_one_parents_children_is_withheld(tmp_path):
    """Review finding E2: five paternal half sibs are five families, and their mean kinship of 0.125 is one man's.

    The related pairs of depth 1 all go through one shared parent, so the
    mean kinship, the group coancestry and the Ne values from them are
    withheld. Mean F rests on no pair and stays.
    """
    source = run(write_ped(tmp_path / "p.tsv", _one_father_rows("MFMFM")), tmp_path / "out")
    assert source is not None
    assert source[ES]["ne_coancestry"]["estimate"]["mean_theta_per_gen"] == {"0": 0.0, "1": 0.125}
    assert source["gates"]["related_families_by_depth"] == {"0": 0, "1": 1}

    written = load(tmp_path / "out")
    assert written["supports"]["depth"]["bins"] == {"0": 6, "1": 5}
    for name, series in (
        ("ne_coancestry", "mean_theta_per_gen"),
        ("ne_group_coancestry", "mean_group_coancestry_per_gen"),
    ):
        assert written[ES][name] == {"ne": None, "reason": WITHHELD, series: None, "ne_per_gen": None}
    assert written[ES]["ne_inbreeding"]["mean_f_per_gen"] == {"0": 0.0, "1": 0.0}


@pytest.mark.parametrize(("couples", "children", "released"), [(2, 5, False), (5, 2, True)])
def test_e2_related_pairs_need_enough_families_and_enough_shared_parents(tmp_path, couples, children, released):
    """Two couples with five children each are four shared parents and two families; five couples pass on both."""
    rows = [{"id": i + 1, "sex": "MF"[i % 2], "mother": -1, "father": -1} for i in range(2 * couples)]
    rows += [
        {"id": 2 * couples + 1 + c * children + j, "sex": "MF"[j % 2], "mother": 2 * c + 2, "father": 2 * c + 1}
        for c in range(couples)
        for j in range(children)
    ]
    source = run(write_ped(tmp_path / "p.tsv", rows), tmp_path / "out", "--estimators", "ne_coancestry")
    assert source is not None
    assert source["gates"]["related_families_by_depth"]["1"] == couples
    assert (load(tmp_path / "out")[ES]["ne_coancestry"].get("reason") != WITHHELD) is released


def test_e2_contribution_ne_is_not_a_moment_of_few_founders_offspring_counts(tmp_path):
    """Review finding E2: with founders and their children only, the contribution Ne is 8 n^2 / sum of k^2.

    Six founders, five children and ``ne: 6.6667`` give a sum of squared
    offspring counts of 30, which only 5, 1, 1, 1, 1, 1 has: one founder is
    a parent of every child. The estimate needs every offspring count to
    be shared by the minimum cell of founders.
    """
    source = run(
        write_ped(tmp_path / "p.tsv", _one_father_rows("MFMFM")),
        tmp_path / "out",
        "--estimators",
        "ne_long_term_contributions",
    )
    assert source is not None
    assert source[ES]["ne_long_term_contributions"]["estimate"]["ne"] == pytest.approx(8 * 5**2 / 30)
    assert source["gates"]["founders_by_children_in_deepest_depth"] == {"1": 5, "5": 1}
    assert load(tmp_path / "out")[ES]["ne_long_term_contributions"] == {"ne": None, "reason": WITHHELD}


def test_e2_contribution_ne_is_released_when_founders_share_their_offspring_counts(tmp_path, wide):
    """Five couples with two children each: every founder has two children, and the estimate is released."""
    rows = [{"id": i + 1, "sex": "MF"[i % 2], "mother": -1, "father": -1} for i in range(10)]
    rows += [
        {"id": 11 + i, "sex": "MF"[i % 2], "mother": 2 * (i // 2) + 2, "father": 2 * (i // 2) + 1} for i in range(10)
    ]
    source = run(write_ped(tmp_path / "p.tsv", rows), tmp_path / "out", "--estimators", "ne_long_term_contributions")
    assert source is not None
    assert source["gates"]["founders_by_children_in_deepest_depth"] == {"2": 10}
    assert load(tmp_path / "out")[ES]["ne_long_term_contributions"] == {"ne": 20.0}

    deep, written, _ = wide
    assert deep["gates"]["founders_by_children_in_deepest_depth"] == {"0": 12}
    assert written[ES]["ne_long_term_contributions"]["ne"] is not None


def test_e2_group_coancestry_does_not_count_a_depths_mz_cotwins(tmp_path):
    """Among unrelated individuals the group coancestry is 1 / (2 x genomes): 12 founders and 0.0455 are one MZ pair."""
    rows = [{"id": i + 1, "sex": "MF"[i % 2], "mother": -1, "father": -1, "birth_year": 1900} for i in range(12)]
    rows += [
        {"id": 13 + i, "sex": "MF"[i % 2], "mother": 2 * (i % 6) + 2, "father": 2 * (i % 6) + 1, "birth_year": 1930}
        for i in range(12)
    ]
    graph = _graph(rows, {1: 3})
    records = compute_effective_size(graph, ALL_EFFECTIVE_SIZE_ESTIMATORS)
    assert records["ne_group_coancestry"]["mean_group_coancestry_per_gen"][0] == pytest.approx(1 / (2 * 11))
    source = build_source(
        meta={"input": "p", "command": "c", "version": "v", "generated_at": "t", "n_total": len(rows)},
        status="complete",
        requested=ALL_EFFECTIVE_SIZE_ESTIMATORS,
        results=records,
        pg=graph,
        reference_rows=None,
    )
    assert source["supports"]["depth"] == {"0": 12, "1": 12}
    assert source["gates"]["genomes_by_depth"] == {"0": 11, "1": 12}
    assert source["gates"]["cotwins_by_depth"] == {"0": 1, "1": 0}
    written = safe_payload(source, DEFAULT_MIN_CELL)
    assert written[ES]["ne_group_coancestry"]["mean_group_coancestry_per_gen"] is None
    assert written[ES]["ne_group_coancestry"]["reason"] == WITHHELD
    assert written[ES]["ne_coancestry"]["mean_theta_per_gen"] is not None

    source["gates"]["cotwins_by_depth"] = {"0": 0, "1": 0}
    assert (
        safe_payload(source, DEFAULT_MIN_CELL)[ES]["ne_group_coancestry"]["mean_group_coancestry_per_gen"] is not None
    )


def test_an_estimate_with_no_count_behind_it_is_withheld(wide):
    """A source that lacks a support (a stop before the counts were taken) withholds every estimate."""
    source, _, _ = wide
    edited = copy.deepcopy(source) | {"supports": {}, "gates": {}}
    records = safe_payload(edited, DEFAULT_MIN_CELL)[ES]
    assert all(record["ne"] is None and record["reason"] == WITHHELD for record in records.values())
    assert all(
        value is None or key in ("reason", "collapses_to_ne_v", "vk_scaled")
        for record in records.values()
        for key, value in record.items()
    )


def test_no_estimate_keeps_its_code_only_when_its_counts_pass(tmp_path):
    """An estimator that ran without an estimate says why when its supports are intact, and is withheld otherwise."""
    for founders, released in ((12, True), (8, False)):
        rows = [
            row | {"sex": "F"} if row["mother"] != -1 else row for row in wide_rows(founders=founders, generations=1)
        ]
        out_dir = tmp_path / f"out{founders}"
        ped = write_ped(tmp_path / f"p{founders}.tsv", rows)
        source = run(ped, out_dir, "--estimators", "ne_variance_family_size")
        assert source is not None
        assert source[ES]["ne_variance_family_size"]["estimate"]["code"] == "single_sex_offspring"
        record = load(out_dir)[ES]["ne_variance_family_size"]
        if released:
            assert (record["ne"], record["reason"], record["code"]) == (None, "no_estimate", "single_sex_offspring")
        else:
            assert record == {"ne": None, "reason": WITHHELD, "ne_per_transition": None}

    source = run(
        write_ped(tmp_path / "q.tsv", wide_rows(generations=1)), tmp_path / "out2", "--estimators", "ne_inbreeding"
    )
    assert source is not None
    record = load(tmp_path / "out2")[ES]["ne_inbreeding"]
    assert (record["ne"], record["reason"], record["code"]) == (None, "no_estimate", "too_few_cohorts")


# ---------------------------------------------------------------------------
# Options that change which rows count
# ---------------------------------------------------------------------------


def test_estimator_selection_limits_the_supports(wide_ped, tmp_path):
    """Only the counts a selected estimator rests on are taken; the others report not_requested."""
    source = run(wide_ped, tmp_path, *BY, "--estimators", "ne_coancestry")
    assert source is not None
    assert set(source["supports"]) == {"depth"}
    assert set(source["gates"]) == {"genomes_by_depth", "related_families_by_depth"}
    written = load(tmp_path)
    assert set(written["supports"]) == {"depth"}
    assert written[ES]["ne_coancestry"]["ne"] is not None
    others = {name: record for name, record in written[ES].items() if name != "ne_coancestry"}
    assert all(record == {"ne": None, "reason": "not_requested"} for record in others.values())


def test_missing_metadata_reports_its_code_and_no_field(tmp_path):
    """An estimator the input cannot support says which column it lacks; the error's row counts are not copied."""
    rows = [row | {"birth_year": -1} if row["mother"] == -1 else row for row in wide_rows(generations=1)]
    out_dir = tmp_path / "out"
    source = run(write_ped(tmp_path / "p.tsv", rows), out_dir, *BY, "--estimators", "ne_hill_overlapping")
    assert source is not None
    run(tmp_path / "p.tsv", tmp_path / "plain", *BY, "--estimators", "ne_hill_overlapping", safe=False)
    ordinary = load(tmp_path / "plain")[ES]["ne_hill_overlapping"]
    assert ordinary["reason"] == "missing_metadata"
    record = load(out_dir)[ES]["ne_hill_overlapping"]
    assert record == {"ne": None, "reason": "missing_metadata", "code": ordinary["code"]}


def test_reference_col_sets_the_reference_supports(wide_ped, tmp_path):
    """Under --reference-col the marked individuals are the support: their count is released and gates the Ne."""
    flags = ("--estimators", "ne_individual_delta_f", "--reference-col", "ref")
    source = run(wide_ped, tmp_path / "all", *flags)
    assert source is not None
    marked = sum(row["ref"] for row in wide_rows())
    assert source["supports"]["reference"] == {"used": marked, "not_used": source["n_total"] - marked}
    assert source["gates"]["reference_used"] == marked == 80
    written = load(tmp_path / "all")
    assert written["supports"]["reference"]["bins"]["used"] == marked
    record = written[ES]["ne_individual_delta_f"]
    assert record["ne"] is not None
    assert record["reference_column"] == "ref"

    rows = wide_rows()
    last = [row for row in rows if row["ref"]][-3:]
    few = [row | {"ref": int(row in last)} for row in rows]
    source = run(write_ped(tmp_path / "few.tsv", few), tmp_path / "few", *flags)
    assert source is not None
    assert source["supports"]["reference"]["used"] == 3
    written = load(tmp_path / "few")
    assert written["supports"]["reference"] == {"status": "blanked", "total": 332}
    record = written[ES]["ne_individual_delta_f"]
    assert (record["ne"], record["reason"], record["standard_error"]) == (None, WITHHELD, None)
    assert record["ne_per_gen"] is not None, "the per-depth Ne does not use the reference"
    assert small_counts(written, DEFAULT_MIN_CELL) == []


@pytest.mark.parametrize("value", ["2", "0", "-1", "3.5", "x"])
def test_min_cell_below_three_is_a_usage_error(tmp_path, value):
    """``--min-cell`` refuses anything but an integer of at least 3, and writes nothing."""
    result = run_pedsum(
        [
            *("effective-size", "--in", str(ADVERSARIAL), "--out", str(tmp_path / "out")),
            "--safe-attempt",
            "--min-cell",
            value,
        ]
    )
    assert result.returncode == 2
    assert "--min-cell" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("min_cell", [3, 10, 20])
def test_min_cell_sets_the_gate_and_is_recorded(tmp_path, min_cell):
    """The file is that of the requested minimum cell, holds no count below it, and says which it was."""
    source = run(ADVERSARIAL, tmp_path, *BY, "--min-cell", str(min_cell))
    assert source is not None
    written = load(tmp_path)
    assert written["safe_attempt_min_cell"] == min_cell
    expected = safe_payload(source, min_cell)
    assert written["supports"] == expected["supports"]
    assert {name: record["ne"] is None for name, record in written[ES].items()} == {
        name: record["ne"] is None for name, record in expected[ES].items()
    }
    assert small_counts(written, min_cell) == []
    if min_cell != 10:
        assert expected["supports"] != safe_payload(source, DEFAULT_MIN_CELL)["supports"]


def test_stop_at_the_memory_limit_goes_through_the_release_list(tmp_path, monkeypatch):
    """The partial file of a safe run is laid out from the release: no estimate, no count, and the stop's reason."""
    rss = [0]
    exited = threading.Event()
    exits: list[int] = []

    def record_exit(code: int) -> None:
        exits.append(code)
        exited.set()

    real_compute = cli.compute_effective_size

    def compute_waits_for_breach(pg, names):
        out = real_compute(pg, names)
        rss[0] = 2 * GiB
        assert exited.wait(timeout=5)
        return out

    monkeypatch.setattr(cli, "compute_effective_size", compute_waits_for_breach)
    args = parse_args(["effective-size", "--in", str(ADVERSARIAL), "--out", str(tmp_path / "out"), "--safe-attempt"])
    with MemoryWatchdog(GiB, poll_s=0.01, rss_reader=lambda: rss[0], exit_fn=record_exit) as wd:
        assert cli.run_effective_size(args, "cmd", wd) == 0
    assert exits == [EXIT_MEMORY_LIMIT]
    assert [path.name for path in (tmp_path / "out").iterdir()] == ["effective_size.txt"]
    written = load(tmp_path / "out")
    assert written["status"] == "stopped_memory_limit"
    assert written["safe_attempt_min_cell"] == DEFAULT_MIN_CELL
    assert written["supports"] == {}
    assert all(
        record == {"ne": None, "reason": "memory_limit", "rss_gib": 2.0, "limit_gib": 1.0}
        for record in written[ES].values()
    )


# ---------------------------------------------------------------------------
# Whole arrays, and the estimates that invert their counts
# ---------------------------------------------------------------------------


def test_hill_cohort_size_mean_does_not_recover_a_blanked_cohort(plain, tmp_path):
    """Blanking small entries of ``N1_m_per_cohort`` beside their mean ``N1_m`` gives the entry back; the engine drops both."""
    min_cell = 20
    hill = plain[ES]["ne_hill_overlapping"]
    males, mean, k = hill["N1_m_per_cohort"], hill["N1_m"], hill["n_eligible_cohorts"]
    assert [count for count in males if count < min_cell] == [15]

    def naive_system() -> System:
        system = System()
        for i, count in enumerate(males):
            if count < min_cell:
                system.var(i, 1, min_cell - 1)
            else:
                system.var(i)
                system.fix(i, count)
        return system

    assert naive_system().pinned([0]) == [], "a blanked entry alone has min_cell - 1 values"
    naive = naive_system()
    naive.add(dict.fromkeys(range(k), 1), k * (mean - HALF_WIDTH), k * (mean + HALF_WIDTH))
    assert naive.pinned([0]) == [0]
    assert naive.value_range(0) == (15, 15)

    source = run(ADVERSARIAL, tmp_path, *BY, "--min-cell", str(min_cell))
    assert source is not None
    written = load(tmp_path)
    assert written["supports"]["birth_cohort_by_sex"] == {"status": "blanked", "total": 207}
    record = written[ES]["ne_hill_overlapping"]
    assert record["reason"] == WITHHELD
    assert all(record[key] is None for key in ("ne", "Ne_m", "Ne_f", "ne_per_cohort", "Ne_m_per_cohort"))
    assert "N1_" not in body(written)
    system, hidden = sex_split_system(source, min_cell)
    cohort_cells = [cell for cell in hidden if cell[0] == COHORT_SEX]
    assert len(cohort_cells) == 2 * (k + 1)
    assert system.pinned(cohort_cells) == []
    assert pinned_sensitive(source, min_cell) == []


def test_sex_ratio_ne_does_not_narrow_a_withheld_sex_split(plain, safe):
    """The sex-ratio Ne of a depth and the depth's size leave one split of its sexes; the engine releases neither."""
    min_cell = 3
    sex_ratio = plain[ES]["ne_sex_ratio"]
    n_male, n_female, ne = (
        sex_ratio["n_male_per_gen"][-1],
        sex_ratio["n_female_per_gen"][-1],
        sex_ratio["ne_per_gen"][-1],
    )
    assert (n_male, n_female) == (3, 1)
    n = n_male + n_female

    source, _ = safe
    system, hidden = sex_split_system(source, min_cell)
    female = (DEPTH_SEX, ("5", "female"))
    assert female in hidden
    assert system.value_range(female) == (0, n), "the released depth size alone leaves every split"

    matches = close_to(ne, HALF_WIDTH)
    add_nonlinear(
        system,
        [(DEPTH_SEX, ("5", "male")), female],
        [(m, n - m) for m in range(n + 1)],
        lambda m, f: m > 0 and f > 0 and matches(4 * m * f / (m + f)),
    )
    assert system.value_range(female) == (1, 3), "with the Ne, the smaller sex has one member"

    written = safe_payload(source, min_cell)
    assert written["supports"]["depth"]["status"] == "intact"
    assert written["supports"]["depth_by_sex"] == {"status": "blanked", "total": 207}
    assert written[ES]["ne_sex_ratio"] == {"ne": None, "reason": WITHHELD, "ne_per_gen": None}
    assert pinned_sensitive(source, min_cell) == []


def test_sex_ratio_ne_is_a_function_of_the_released_table(wide):
    """When released, each depth's sex-ratio Ne is ``4 m f / (m + f)`` of the released cells: it adds no number."""
    _, written, _ = wide
    cells = written["supports"]["depth_by_sex"]["cells"]
    per_depth = written[ES]["ne_sex_ratio"]["ne_per_gen"]
    for depth, ne in per_depth.items():
        m, f = cells[depth]["male"], cells[depth]["female"]
        assert ne == pytest.approx(4 * m * f / (m + f), abs=HALF_WIDTH)
    finite = np.array(list(per_depth.values()))
    assert written[ES]["ne_sex_ratio"]["ne"] == pytest.approx(len(finite) / (1 / finite).sum(), abs=HALF_WIDTH)


# ---------------------------------------------------------------------------
# Exact oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_cell", [3, DEFAULT_MIN_CELL, 10, 20])
def test_oracle_adversarial_and_wide_pedigrees(safe, wide, min_cell):
    """No protected cell of either pedigree's release is pinned, at any minimum cell."""
    assert pinned_sensitive(safe[0], min_cell) == []
    assert pinned_sensitive(wide[0], min_cell) == []


def _write_generated(frame: pl.DataFrame, path: Path) -> Path:
    """Write a generated frame as a pedigree with every sex resolved and birth years that follow depth.

    A parent link is dropped when the parent already has the other role or
    the other sex; an unknown sex becomes the parent's role, else alternates.
    """
    sex = frame["sex"].to_list()
    role: dict[int, int] = {}

    def parent(row: int, as_sex: int) -> int:
        other = SEX_MALE if as_sex == SEX_FEMALE else SEX_FEMALE
        if row == -1 or sex[row] == other or role.setdefault(row, as_sex) != as_sex:
            return -1
        return row + 1

    links = [
        (parent(mother, SEX_FEMALE), parent(father, SEX_MALE))
        for mother, father in frame.select("mother", "father").iter_rows()
    ]
    depth: list[int] = []
    for mother, father in links:
        depth.append(1 + max((depth[p - 1] for p in (mother, father) if p != -1), default=-1))
    rows = []
    for i, (mother, father) in enumerate(links):
        resolved = role.get(i, sex[i] if sex[i] in (SEX_MALE, SEX_FEMALE) else (SEX_MALE, SEX_FEMALE)[i % 2])
        rows.append(
            {
                "id": i + 1,
                "sex": "M" if resolved == SEX_MALE else "F",
                "mother": mother,
                "father": father,
                "birth_year": 1300 + 10 * depth[i] + i % 2,
            }
        )
    return write_ped(path, rows)


@pytest.mark.slow
@pytest.mark.filterwarnings(r"ignore:.*pg\.sex is uniform:RuntimeWarning")
@property_budget(40)
@given(frame=acyclic_pedigree_frames(max_size=70).filter(lambda frame: len(frame) > 0))
def test_oracle_generated_pedigrees(frame):
    """On generated pedigrees no protected cell is pinned and no file holds a small count, at minimum cell 3 and 5."""
    with tempfile.TemporaryDirectory() as tmp:
        ped = _write_generated(frame, Path(tmp) / "p.tsv")
        for min_cell in (3, DEFAULT_MIN_CELL):
            out_dir = Path(tmp) / f"out{min_cell}"
            source = run(ped, out_dir, *BY, "--birth-year-min", "1000", "--min-cell", str(min_cell))
            assert source is not None
            assert pinned_sensitive(source, min_cell) == []
            assert small_counts(load(out_dir), min_cell) == []


@property_budget(30)
@given(
    seed=st.integers(0, 10_000),
    founders=st.integers(4, 16),
    per_generation=st.integers(6, 40),
    generations=st.integers(1, 4),
)
def test_oracle_random_mating_pedigrees(seed, founders, per_generation, generations):
    """On pedigrees large enough that some estimates are released and some are not, no protected cell is pinned."""
    with tempfile.TemporaryDirectory() as tmp:
        ped = write_ped(Path(tmp) / "p.tsv", wide_rows(seed, founders, per_generation, generations))
        for min_cell in (3, DEFAULT_MIN_CELL):
            out_dir = Path(tmp) / f"out{min_cell}"
            source = run(ped, out_dir, *BY, "--reference-col", "ref", "--min-cell", str(min_cell))
            assert source is not None
            assert pinned_sensitive(source, min_cell) == []
            assert small_counts(load(out_dir), min_cell) == []


def test_oracle_catches_a_sex_table_released_cell_by_cell(safe):
    """With the depth table's small tail merged away and one sex released per depth, the merged depths are pinned.

    The engine withholds the depth-by-sex table whole. A reader given each
    depth's males anyway could not be told the depth sizes the merge hides.
    """
    source, _ = safe
    spec = effective_size_spec(source)
    protected = released_tables(source, spec, DEFAULT_MIN_CELL)
    assert protected[DEPTH].status.value == "merged"
    assert protected[DEPTH_SEX].status.value == "blanked"
    system, hidden = system_from_release(spec, protected, DEFAULT_MIN_CELL)
    merged = [cell for cell in hidden if cell[0] == DEPTH]
    assert [label for _, label in merged] == ["4", "5"]
    assert system.pinned(merged) == []
    for depth, cells in at(source, DEPTH_SEX).items():
        for sex_label, count in cells.items():
            system.fix((DEPTH_SEX, (depth, sex_label)), count)
    assert system.pinned(merged) == merged
