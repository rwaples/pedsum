"""The ``effective-size`` subcommand (ADR 0004): selection, the merge rule, and partial output at the memory limit."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time

import pytest
import yaml
from conftest import EXAMPLE, REPO, run_pedsum, write_ped, write_stripped_pedigree
from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS

from pedsum import cli
from pedsum.memory import EXIT_MEMORY_LIMIT, GiB, MemoryWatchdog

COANCESTRY = ("ne_coancestry", "ne_group_coancestry")


def _run_es(tmp_path, *extra, pedigree=EXAMPLE):
    out_dir = tmp_path / "out"
    res = run_pedsum(["effective-size", "--in", str(pedigree), "--out", str(out_dir), *extra])
    return res, out_dir / "effective_size.yaml"


def _load(path) -> dict:
    return yaml.safe_load(path.read_text())


def _values(data: dict) -> dict:
    return {name: record["ne"] for name, record in data["effective_size"].items() if record["ne"] is not None}


def _reasons(data: dict) -> dict:
    return {name: record.get("reason") for name, record in data["effective_size"].items()}


def test_default_runs_all_eight_estimators(tmp_path):
    """The default selection fills all eight estimators."""
    res, path = _run_es(tmp_path)
    assert res.returncode == 0, res.stderr
    data = _load(path)
    assert data["status"] == "complete"
    assert data["estimators_requested"] == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
    assert list(data["effective_size"]) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
    assert set(_values(data)) == set(ALL_EFFECTIVE_SIZE_ESTIMATORS)
    assert data["n_total"] == 200
    assert {"input", "command", "version", "generated_at"} <= set(data)


def test_all_is_the_default(tmp_path):
    """``--estimators all`` writes the same records as the default run."""
    res, path = _run_es(tmp_path / "default")
    assert res.returncode == 0, res.stderr
    default = _load(path)["effective_size"]
    res, path = _run_es(tmp_path / "all", "--estimators", "all")
    assert res.returncode == 0, res.stderr
    assert _load(path)["effective_size"] == default


def test_single_named_estimator(tmp_path):
    """One named estimator runs; the other seven are not_requested."""
    res, path = _run_es(tmp_path, "--estimators", "ne_inbreeding")
    assert res.returncode == 0, res.stderr
    data = _load(path)
    assert data["estimators_requested"] == ["ne_inbreeding"]
    assert isinstance(data["effective_size"]["ne_inbreeding"]["ne"], float)
    reasons = _reasons(data)
    assert all(reasons[name] == "not_requested" for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name != "ne_inbreeding")


def test_repeated_and_comma_flags_extend(tmp_path):
    """Repeated ``--estimators`` flags and comma lists combine, in canonical order."""
    res, path = _run_es(tmp_path, "--estimators", "ne_sex_ratio", "--estimators", "ne_coancestry,ne_inbreeding")
    assert res.returncode == 0, res.stderr
    assert _load(path)["estimators_requested"] == ["ne_inbreeding", "ne_coancestry", "ne_sex_ratio"]


def test_coancestry_pair_selection(tmp_path):
    """Only the coancestry pair: two values, six not_requested, status complete."""
    res, path = _run_es(tmp_path, "--estimators", ",".join(COANCESTRY))
    assert res.returncode == 0, res.stderr
    data = _load(path)
    assert data["status"] == "complete"
    assert set(_values(data)) == set(COANCESTRY)
    assert all(
        _reasons(data)[name] == "not_requested" for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name not in COANCESTRY
    )


@pytest.mark.parametrize("value", ["ne", "ne_inbreeding,bogus", "all,bogus"])
def test_unknown_estimator_exits_2(tmp_path, value):
    """An unknown name is a usage error that lists the choices."""
    res, path = _run_es(tmp_path, "--estimators", value)
    assert res.returncode == 2
    assert "unknown estimator" in res.stderr
    assert "ne_group_coancestry" in res.stderr
    assert not path.exists()


def test_hill_collapses_without_birth_year(tmp_path):
    """Without ``--birth-year-col`` Ne_H collapses to Ne_V."""
    pedigree = write_stripped_pedigree(tmp_path / "no_birth_year.tsv")
    res, path = _run_es(tmp_path, pedigree=pedigree)
    assert res.returncode == 0, res.stderr
    hill = _load(path)["effective_size"]["ne_hill_overlapping"]
    assert hill["collapses_to_ne_v"] is True
    assert hill["cohort_window"] is None
    assert hill["Ne_m"] is None
    assert hill["Ne_f"] is None
    assert hill["n_eligible_cohorts"] == 0


def test_hill_populated_with_birth_year(tmp_path):
    """``--birth-year-col`` gives Ne_H its cohort window and sex-decomposed scalars."""
    res, path = _run_es(tmp_path, "--birth-year-col", "birth_year")
    assert res.returncode == 0, res.stderr
    hill = _load(path)["effective_size"]["ne_hill_overlapping"]
    assert hill["collapses_to_ne_v"] is False
    assert hill["cohort_window"]["c_min"] <= hill["cohort_window"]["c_max"]
    assert hill["n_eligible_cohorts"] >= 1
    assert isinstance(hill["Ne_m"], float)
    assert isinstance(hill["Ne_f"], float)
    assert hill["generation_interval"] > 0


def test_birth_year_missing_column_errors(tmp_path):
    """Naming a non-existent birth-year column fails validation."""
    res, path = _run_es(tmp_path, "--birth-year-col", "does_not_exist")
    assert res.returncode == 1
    assert "does_not_exist" in res.stderr
    assert not path.exists()


def test_observed_depth_labels_index_the_arrays(tmp_path):
    """Each record carries the depth labels it observed; rate arrays are transition-aligned."""
    res, path = _run_es(tmp_path, "--estimators", "all")
    assert res.returncode == 0, res.stderr
    es = _load(path)["effective_size"]
    for name in ("ne_inbreeding", "ne_coancestry", "ne_group_coancestry"):
        depths = es[name]["depths"]
        assert depths == sorted(set(depths))
        assert len(es[name]["transition_from"]) == len(depths) - 1
        assert len(es[name]["ne_per_gen"]) == len(depths) - 1
    assert len(es["ne_variance_family_size"]["ne_per_transition"]) == len(
        es["ne_variance_family_size"]["parent_depths"]
    )
    assert len(es["ne_sex_ratio"]["ne_per_gen"]) == len(es["ne_sex_ratio"]["depths"])


def test_unknown_sex_is_refused(tmp_path):
    """Rows left at unknown sex under ``--allow-missing-sex`` stop the command with exit 1."""
    pedigree = write_ped(
        tmp_path / "ped.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 8, "sex": "", "mother": -1, "father": -1},
            {"id": 3, "sex": "M", "mother": 2, "father": 1},
        ],
    )
    res, path = _run_es(tmp_path, "--allow-missing-sex", pedigree=pedigree)
    assert res.returncode == 1
    assert "effective size needs resolved sex" in res.stderr
    assert not path.exists()


_BREACH_SCRIPT = """
import sys, time
import numpy as np
from pedsum import cli
from pedsum.memory import read_rss_bytes

real = cli.compute_effective_size

def estimators_allocate(pg, names):
    block = np.ones(600 * 2**20, dtype=np.uint8)
    time.sleep(30)
    return real(pg, names)

cli.compute_effective_size = estimators_allocate
limit = read_rss_bytes() + 300 * 2**20
sys.exit(cli.main([*sys.argv[1:], "--max-memory", str(limit)]))
"""


def _breach_in_subprocess(tmp_path, *estimators):
    out_dir = tmp_path / "out"
    argv = ["effective-size", "--in", str(EXAMPLE), "--out", str(out_dir), "--estimators", *estimators]
    res = subprocess.run(
        [sys.executable, "-c", _BREACH_SCRIPT, *argv],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return res, out_dir / "effective_size.yaml"


def test_breach_while_estimating_marks_every_requested_estimator(tmp_path):
    """A stop in the estimator step publishes every requested estimator as memory_limit."""
    res, path = _breach_in_subprocess(tmp_path, "all")
    assert res.returncode == EXIT_MEMORY_LIMIT, res.stderr
    assert f"effective size ({', '.join(ALL_EFFECTIVE_SIZE_ESTIMATORS)}) used" in res.stderr
    data = _load(path)
    assert data["status"] == "stopped_memory_limit"
    for name in ALL_EFFECTIVE_SIZE_ESTIMATORS:
        record = data["effective_size"][name]
        assert record["ne"] is None
        assert record["reason"] == "memory_limit"
        assert record["rss_gib"] > record["limit_gib"]
    assert not [p for p in path.parent.iterdir() if ".partial-" in p.name]


def test_breach_with_a_single_estimator(tmp_path):
    """Only the requested estimator is memory_limit; the other seven stay not_requested."""
    res, path = _breach_in_subprocess(tmp_path, "ne_coancestry")
    assert res.returncode == EXIT_MEMORY_LIMIT, res.stderr
    data = _load(path)
    assert data["status"] == "stopped_memory_limit"
    reasons = _reasons(data)
    assert reasons["ne_coancestry"] == "memory_limit"
    assert all(reasons[name] == "not_requested" for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name != "ne_coancestry")


_LOAD_BREACH_SCRIPT = """
import sys, time
import numpy as np
from pedsum import cli
from pedsum.memory import read_rss_bytes

real = cli.load_and_validate

def load_allocates(*args, **kwargs):
    block = np.ones(600 * 2**20, dtype=np.uint8)
    time.sleep(30)
    return real(*args, **kwargs)

cli.load_and_validate = load_allocates
limit = read_rss_bytes() + 300 * 2**20
sys.exit(cli.main([*sys.argv[1:], "--max-memory", str(limit)]))
"""


def test_breach_while_loading_replaces_an_earlier_complete_file(tmp_path):
    """A stop before the graph exists still writes this run's file, not a stale complete one."""
    res, path = _run_es(tmp_path)
    assert res.returncode == 0, res.stderr
    assert _load(path)["status"] == "complete"

    argv = ["effective-size", "--in", str(EXAMPLE), "--out", str(path.parent)]
    res = subprocess.run(
        [sys.executable, "-c", _LOAD_BREACH_SCRIPT, *argv],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert res.returncode == EXIT_MEMORY_LIMIT, res.stderr
    assert "load+validate used" in res.stderr
    data = _load(path)
    assert data["status"] == "stopped_memory_limit"
    assert data["n_total"] is None
    assert _values(data) == {}
    assert set(_reasons(data).values()) == {"memory_limit"}


def _args(tmp_path, *extra):
    return cli._parse_args(["effective-size", "--in", str(EXAMPLE), "--out", str(tmp_path / "out"), *extra])


def test_breach_during_final_publication_keeps_the_complete_file(tmp_path, monkeypatch, caplog):
    """The main thread holds the publication lock first: the file is complete and nothing exits."""
    rss = [0]
    sampled_high = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    exits: list[int] = []
    codes: list[int] = []

    def reader() -> int:
        if rss[0]:
            sampled_high.set()
        return rss[0]

    real_write = cli._write_yaml

    def slow_write(data, path):
        if data["status"] == "complete":
            entered.set()
            assert release.wait(timeout=30)
        real_write(data, path)

    monkeypatch.setattr(cli, "_write_yaml", slow_write)
    args = _args(tmp_path)
    wd = MemoryWatchdog(GiB, poll_s=0.01, rss_reader=reader, exit_fn=exits.append)
    with caplog.at_level(logging.INFO), wd:
        runner = threading.Thread(target=lambda: codes.append(cli._run_effective_size(args, "cmd", wd)))
        runner.start()
        assert entered.wait(timeout=60)
        rss[0] = 2 * GiB
        assert sampled_high.wait(timeout=5)
        time.sleep(0.05)
        release.set()
        runner.join(timeout=30)
        assert wd._thread is not None
        wd._thread.join(timeout=5)
    assert codes == [0]
    assert exits == []
    assert _load(args.out_dir / "effective_size.yaml")["status"] == "complete"
    assert any("after its output was published" in r.getMessage() for r in caplog.records)


def test_breach_before_final_publication_wins(tmp_path, monkeypatch):
    """The breach takes the lock first: the main thread's publish is skipped and the file stays stopped."""
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
    args = _args(tmp_path, "--estimators", "all")
    with MemoryWatchdog(GiB, poll_s=0.01, rss_reader=lambda: rss[0], exit_fn=record_exit) as wd:
        assert cli._run_effective_size(args, "cmd", wd) == 0
    assert exits == [EXIT_MEMORY_LIMIT]
    data = _load(args.out_dir / "effective_size.yaml")
    assert data["status"] == "stopped_memory_limit"
    assert set(_reasons(data).values()) == {"memory_limit"}


@pytest.mark.parametrize("flag", ["--effective-size", "--no-effective-size", "--ne-coancestry"])
def test_removed_summarize_flags_name_the_new_command(tmp_path, flag):
    """The 0.14 Ne flags on ``summarize`` exit 2 and point at ``effective-size`` (ADR 0004 §2)."""
    out_dir = tmp_path / "out"
    res = run_pedsum(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), flag])
    assert res.returncode == 2
    assert f"{flag} was removed in 0.15.0" in res.stderr
    assert "pedsum effective-size --in X --out Y" in res.stderr
    assert not (out_dir / "summary.yaml").exists()


def test_ne_threads_flag_is_gone(tmp_path):
    """``--ne-threads`` was removed; argparse rejects it instead of silently ignoring it."""
    out_dir = tmp_path / "out"
    res = run_pedsum(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--ne-threads", "4"])
    assert res.returncode != 0
    assert "--ne-threads" in res.stderr
    assert not (out_dir / "summary.yaml").exists()


def _row(id_, sex, mother=-1, father=-1, **extra):
    return {"id": id_, "sex": sex, "mother": mother, "father": father, **extra}


#: Tiny pedigrees whose estimators run and find no estimate, one per kind.
#: ``trio``: depths 0 and 1 only. ``outbred``: three depths, every F = 0,
#: mean kinship falling, and every parent with two offspring. ``two_males``:
#: two founders and nothing else. ``late_inbred``: depth 2 is inbred, depth 3
#: (the default reference) is not.
_NO_ESTIMATE_PEDIGREES = {
    "trio": [_row(1, "M"), _row(2, "F"), _row(3, "F", 2, 1)],
    "trio_birth_year": [
        _row(1, "M", birth_year=2000),
        _row(2, "F", birth_year=2000),
        _row(3, "F", 2, 1, birth_year=2020),
    ],
    "outbred": [
        *(_row(i, s) for i, s in ((1, "M"), (2, "F"), (3, "F"), (4, "M"))),
        _row(5, "M", 2, 1),
        _row(6, "F", 2, 1),
        _row(7, "M", 3, 5),
        _row(8, "F", 3, 5),
        _row(9, "M", 6, 4),
        _row(10, "F", 6, 4),
    ],
    "two_males": [_row(1, "M"), _row(2, "M")],
    "late_inbred": [
        _row(1, "M"),
        _row(2, "F"),
        _row(3, "M", 2, 1),
        _row(4, "F", 2, 1),
        _row(5, "M", 4, 3),
        _row(6, "F"),
        _row(7, "F", 6, 5),
    ],
}


def _run_tiny(tmp_path, name, *extra):
    pedigree = write_ped(tmp_path / f"{name}.tsv", _NO_ESTIMATE_PEDIGREES[name])
    if name.endswith("birth_year"):
        extra = ("--birth-year-col", "birth_year", *extra)
    res, path = _run_es(tmp_path / name, *extra, pedigree=pedigree)
    assert res.returncode == 0, res.stderr
    return res, _load(path)["effective_size"]


_UNIFORM_SEX = pytest.mark.filterwarnings(r"ignore:.*pg\.sex is uniform:RuntimeWarning")


@pytest.mark.parametrize(
    ("pedigree", "estimator", "code"),
    [
        ("trio", "ne_inbreeding", "too_few_cohorts"),
        ("trio", "ne_coancestry", "too_few_cohorts"),
        ("trio", "ne_group_coancestry", "too_few_cohorts"),
        ("outbred", "ne_inbreeding", "no_positive_rate"),
        ("outbred", "ne_coancestry", "no_positive_rate"),
        ("outbred", "ne_group_coancestry", "no_positive_rate"),
        ("trio", "ne_individual_delta_f", "reference_not_inbred"),
        ("two_males", "ne_individual_delta_f", "empty_reference"),
        ("trio", "ne_variance_family_size", "too_few_parents"),
        ("outbred", "ne_variance_family_size", "no_family_size_variance"),
        ("two_males", "ne_sex_ratio", "no_depth_with_both_sexes"),
        ("trio", "ne_hill_overlapping", "no_estimable_transition"),
        ("trio_birth_year", "ne_hill_overlapping", "no_eligible_cohorts"),
    ],
)
@_UNIFORM_SEX
def test_no_estimate_carries_its_code(tmp_path, pedigree, estimator, code):
    """An estimator that runs without an estimate says why in ``reason`` and ``code``."""
    _, records = _run_tiny(tmp_path, pedigree)
    record = records[estimator]
    assert (record["ne"], record["reason"], record["code"]) == (None, "no_estimate", code)
    assert "fields" not in record


@_UNIFORM_SEX
def test_every_null_ne_has_a_reason(tmp_path):
    """Drift guard: a null ``ne`` always has a reason and code; an estimate has neither."""
    runs = [_run_tiny(tmp_path, name)[1] for name in _NO_ESTIMATE_PEDIGREES]
    for pedigree in (EXAMPLE, write_stripped_pedigree(tmp_path / "no_birth_year.tsv")):
        res, path = _run_es(tmp_path / pedigree.stem, pedigree=pedigree)
        assert res.returncode == 0, res.stderr
        runs.append(_load(path)["effective_size"])
    res, path = _run_es(tmp_path / "with_birth_year", "--birth-year-col", "birth_year")
    assert res.returncode == 0, res.stderr
    runs.append(_load(path)["effective_size"])
    for records in runs:
        assert list(records) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        for name, record in records.items():
            if record["ne"] is None:
                assert record["reason"] == "no_estimate", (name, record)
                assert isinstance(record["code"], str), (name, record)
            else:
                assert "reason" not in record, (name, record)
                assert "code" not in record, (name, record)


def _run_es_in_process(tmp_path, pedigree, *extra):
    out_dir = tmp_path / "out"
    args = cli._parse_args(["effective-size", "--in", str(pedigree), "--out", str(out_dir), *extra])
    assert cli._run_effective_size(args, "cmd", MemoryWatchdog(None)) == 0
    return _load(out_dir / "effective_size.yaml")["effective_size"]["ne_individual_delta_f"]


def _reference_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.getMessage().startswith("ne_individual_delta_f has no estimate")
    ]


def test_null_reference_warns_when_another_depth_has_an_estimate(tmp_path, caplog):
    """The default reference finds no estimate though an earlier depth has one: warn and name it."""
    pedigree = write_ped(tmp_path / "late_inbred.tsv", _NO_ESTIMATE_PEDIGREES["late_inbred"])
    with caplog.at_level(logging.WARNING):
        record = _run_es_in_process(tmp_path, pedigree)
    assert record["code"] == "reference_not_inbred"
    assert any(ne is not None for ne in record["ne_per_gen"])
    [message] = _reference_warnings(caplog)
    assert "the last observed depth (depth 3)" in message
    assert "n_reference=1" in message
    assert "--reference-col" in message


def test_null_reference_col_warning_names_the_column(tmp_path, caplog):
    """With ``--reference-col`` the warning names the column instead of a depth."""
    rows = [{**row, "ref": int(row["id"] == 7)} for row in _NO_ESTIMATE_PEDIGREES["late_inbred"]]
    pedigree = write_ped(tmp_path / "late_inbred_ref.tsv", rows)
    with caplog.at_level(logging.WARNING):
        _run_es_in_process(tmp_path, pedigree, "--reference-col", "ref")
    [message] = _reference_warnings(caplog)
    assert "the reference column 'ref'" in message


@pytest.mark.parametrize("pedigree", ["trio", "example"])
def test_no_reference_warning_otherwise(tmp_path, caplog, pedigree):
    """No warning when the headline has an estimate, or when no depth has one."""
    path = EXAMPLE if pedigree == "example" else write_ped(tmp_path / "p.tsv", _NO_ESTIMATE_PEDIGREES[pedigree])
    with caplog.at_level(logging.WARNING):
        record = _run_es_in_process(tmp_path, path)
    assert (record["ne"] is None) == (pedigree == "trio")
    assert _reference_warnings(caplog) == []
