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
from pedsum.base import KINSHIP_DP_ESTIMATORS
from pedsum.memory import EXIT_MEMORY_LIMIT, GiB, MemoryWatchdog

DEFAULT_SIX = [name for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name not in KINSHIP_DP_ESTIMATORS]


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


def test_default_runs_the_six_non_dp_estimators(tmp_path):
    """The default selection fills six estimators and leaves the kinship-DP pair not_requested."""
    res, path = _run_es(tmp_path)
    assert res.returncode == 0, res.stderr
    data = _load(path)
    assert data["status"] == "complete"
    assert data["estimators_requested"] == DEFAULT_SIX
    assert list(data["effective_size"]) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
    assert set(_values(data)) == set(DEFAULT_SIX)
    for name in KINSHIP_DP_ESTIMATORS:
        assert data["effective_size"][name]["ne"] is None
        assert data["effective_size"][name]["reason"] == "not_requested"
    assert data["n_total"] == 200
    assert {"input", "command", "version", "generated_at"} <= set(data)


def test_all_keeps_step_one_values_and_adds_the_dp_pair(tmp_path):
    """Merge rule: the six step-1 values under ``all`` equal the default run's, and Ne_C / Ne_GC are floats."""
    res, path = _run_es(tmp_path / "default")
    assert res.returncode == 0, res.stderr
    default = _load(path)["effective_size"]
    res, path = _run_es(tmp_path / "all", "--estimators", "all")
    assert res.returncode == 0, res.stderr
    data = _load(path)
    assert data["estimators_requested"] == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
    for name in DEFAULT_SIX:
        assert data["effective_size"][name] == default[name], name
    for name in KINSHIP_DP_ESTIMATORS:
        assert isinstance(data["effective_size"][name]["ne"], float), name


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


def test_dp_only_selection(tmp_path):
    """Only the DP pair: two values, six not_requested, status complete."""
    res, path = _run_es(tmp_path, "--estimators", "ne_coancestry,ne_group_coancestry")
    assert res.returncode == 0, res.stderr
    data = _load(path)
    assert data["status"] == "complete"
    assert set(_values(data)) == set(KINSHIP_DP_ESTIMATORS)
    assert all(_reasons(data)[name] == "not_requested" for name in DEFAULT_SIX)


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

def dp_allocates(pg, names):
    if any(name in cli.KINSHIP_DP_ESTIMATORS for name in names):
        block = np.ones(600 * 2**20, dtype=np.uint8)
        time.sleep(30)
    return real(pg, names)

cli.compute_effective_size = dp_allocates
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


def test_breach_in_dp_step_keeps_the_six(tmp_path):
    """A stop during the DP step publishes the six finished values and marks the DP pair memory_limit."""
    res, path = _breach_in_subprocess(tmp_path, "all")
    assert res.returncode == EXIT_MEMORY_LIMIT, res.stderr
    assert "effective size (ne_coancestry, ne_group_coancestry) used" in res.stderr
    data = _load(path)
    assert data["status"] == "stopped_memory_limit"
    for name in KINSHIP_DP_ESTIMATORS:
        record = data["effective_size"][name]
        assert record["ne"] is None
        assert record["reason"] == "memory_limit"
        assert record["rss_gib"] > record["limit_gib"]
    assert set(_values(data)) == set(DEFAULT_SIX)
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

    def dp_waits_for_breach(pg, names):
        out = real_compute(pg, names)
        if "ne_coancestry" in names:
            rss[0] = 2 * GiB
            assert exited.wait(timeout=5)
        return out

    monkeypatch.setattr(cli, "compute_effective_size", dp_waits_for_breach)
    args = _args(tmp_path, "--estimators", "all")
    with MemoryWatchdog(GiB, poll_s=0.01, rss_reader=lambda: rss[0], exit_fn=record_exit) as wd:
        assert cli._run_effective_size(args, "cmd", wd) == 0
    assert exits == [EXIT_MEMORY_LIMIT]
    data = _load(args.out_dir / "effective_size.yaml")
    assert data["status"] == "stopped_memory_limit"
    assert {_reasons(data)[name] for name in KINSHIP_DP_ESTIMATORS} == {"memory_limit"}
