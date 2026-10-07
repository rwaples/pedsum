"""The ``assortative-mating`` subcommand: exit codes, file layout, and the YAML contract."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import yaml
from conftest import EXAMPLE, run_pedsum

from pedsum.cli import _parse_args, physical_cores

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def traits_tsv(tmp_path_factory) -> Path:
    """``example_pedigree.tsv`` plus trait columns covering each typing and error path."""
    df = pl.read_csv(EXAMPLE, separator="\t")
    n = len(df)
    rng = np.random.default_rng(0)
    liability = df["liability"].to_numpy()
    df = df.with_columns(
        pl.Series("dx", np.where(liability > 0.5, "yes", "no")),
        pl.Series("score", (liability + rng.normal(size=n)).round(3)),
        pl.Series("with_sentinel", np.where(rng.random(n) < 0.2, "-9", liability.astype(str))),
        pl.Series("all_na", ["NA"] * n),
        pl.Series("const", ["4"] * n),
        pl.Series("bad", ["abc"] + ["1.0"] * (n - 1)),
        pl.Series("infinite", ["inf", *liability[1:].astype(str).tolist()]),
    )
    path = tmp_path_factory.mktemp("am") / "traits.tsv"
    df.write_csv(path, separator="\t")
    return path


def _run(tmp_path, pedigree, *extra) -> tuple[subprocess.CompletedProcess, Path]:
    out_dir = tmp_path / "out"
    res = run_pedsum(["assortative-mating", "--in", str(pedigree), "--out", str(out_dir), *extra])
    return res, out_dir / "assortative_mating.yaml"


def _load(path) -> dict:
    return yaml.safe_load(path.read_text())


def test_one_continuous_trait(tmp_path, traits_tsv):
    """One continuous trait writes the meta block and one computed cell with CI and p-value."""
    res, path = _run(tmp_path, traits_tsv, "--trait", "liability", "--permutations", "99", "--bootstrap", "99")
    assert res.returncode == 0, res.stderr
    assert sorted(p.name for p in path.parent.iterdir()) == ["assortative_mating.yaml"]
    data = _load(path)
    assert {"input", "command", "version", "generated_at", "n_total"} <= set(data)
    am = data["assortative_mating"]
    assert list(am) == ["traits", "settings", "inference", "mating_pairs", "mate_correlation", "notes"]
    assert am["traits"] == [
        {"name": "liability", "type": "continuous", "type_source": "inferred", "n_values": 200, "n_missing": 0}
    ]
    assert am["inference"]["bootstrap_unit"] == "mate_network"
    (cell,) = am["mate_correlation"]
    pearson = cell["crude"]["pearson"]
    assert -1 <= pearson["r"] <= 1
    assert pearson["ci"][0] <= pearson["ci"][1]
    assert pearson["ci_method"] == "bootstrap"
    assert pearson["se"] > 0
    assert 0 < pearson["p_perm"] <= 1
    assert pearson["permutations"]["requested"] == 99
    assert "p_perm" not in cell["crude"]["spearman"]


def test_two_traits_report_every_cell(tmp_path, traits_tsv):
    """Two traits give four cells, each with its kind pair's primary estimator, and a within-person block."""
    res, path = _run(tmp_path, traits_tsv, "--trait", "liability", "dx", "--permutations", "9", "--bootstrap", "9")
    assert res.returncode == 0, res.stderr
    am = _load(path)["assortative_mating"]
    assert am["traits"][1] == {
        "name": "dx",
        "type": "binary",
        "type_source": "inferred",
        "levels": ["no", "yes"],
        "n_values": 200,
        "n_missing": 0,
    }
    cells = [(c["mother"], c["father"], list(c["crude"])) for c in am["mate_correlation"]]
    assert cells == [
        ("liability", "liability", ["pearson", "spearman"]),
        ("liability", "dx", ["biserial", "point_biserial"]),
        ("dx", "liability", ["biserial", "point_biserial"]),
        ("dx", "dx", ["table", "tetrachoric", "odds_ratio", "phi"]),
    ]
    binary = am["mate_correlation"][3]["crude"]
    assert sum(map(sum, binary["table"])) == am["mate_correlation"][3]["n"]
    assert {"rho", "boundary", "se", "ci", "ci_method", "p_perm", "bootstrap", "permutations"} <= set(
        binary["tetrachoric"]
    )
    assert "p_perm" not in binary["odds_ratio"]
    for sex in ("mothers", "fathers"):
        assert am["within_person"][sex]["estimator"] == "biserial"
        assert -1 <= am["within_person"][sex]["rho"] <= 1


def test_default_inference_is_the_sandwich(tmp_path, traits_tsv):
    """Without --bootstrap the CI is the Wald interval from the sandwich SE; --permutations 0 turns the p-value off."""
    res, path = _run(tmp_path, traits_tsv, "--trait", "liability", "--permutations", "0")
    assert res.returncode == 0, res.stderr
    am = _load(path)["assortative_mating"]
    assert (am["settings"]["bootstrap"], am["settings"]["ci_method"]) == (0, "sandwich")
    assert "se_method" in am["inference"]
    crude = am["mate_correlation"][0]["crude"]
    pearson = crude["pearson"]
    assert pearson["se"] > 0
    assert pearson["ci"][0] < pearson["r"] < pearson["ci"][1]
    assert (pearson["ci_method"], pearson["ci_unavailable_reason"]) == ("sandwich", None)
    assert "bootstrap" not in pearson
    assert (pearson["p_perm"], pearson["p_perm_unavailable_reason"]) == (None, "not_requested")
    spearman = crude["spearman"]
    assert (spearman["se"], spearman["ci"], spearman["ci_unavailable_reason"]) == (
        None,
        None,
        "bootstrap_not_requested",
    )


def test_trait_missing_and_stated_type(tmp_path, traits_tsv):
    """--trait-missing removes the sentinel before typing; --trait-type records type_source: stated."""
    res, path = _run(
        tmp_path,
        traits_tsv,
        "--trait",
        "with_sentinel",
        "--trait-missing",
        "-9",
        "--trait-type",
        "with_sentinel=continuous",
        "--permutations",
        "9",
        "--bootstrap",
        "9",
    )
    assert res.returncode == 0, res.stderr
    (trait,) = _load(path)["assortative_mating"]["traits"]
    assert trait["type_source"] == "stated"
    assert trait["n_missing"] > 0


def test_two_continuous_traits_stratified_by_depth(tmp_path, traits_tsv):
    """Two continuous traits with --stratify-by depth report a stratified pearson per cell and the within-person r."""
    res, path = _run(
        tmp_path,
        traits_tsv,
        "--trait",
        "liability",
        "score",
        "--stratify-by",
        "depth",
        "--permutations",
        "19",
        "--bootstrap",
        "19",
    )
    assert res.returncode == 0, res.stderr
    am = _load(path)["assortative_mating"]
    assert am["settings"]["stratify_by"] == "depth"
    assert list(am["mating_pairs"]["n_dropped"]) == ["unknown_stratum"]
    for cell in am["mate_correlation"]:
        stratified = cell["stratified"]["pearson"]
        assert -1 <= stratified["r"] <= 1
        assert stratified["n_strata_mothers"] >= 1
        assert stratified["permutations"]["requested"] == 19
        assert "degenerate_stratum" in cell["n_dropped"]
    for sex in ("mothers", "fathers"):
        assert am["within_person"][sex]["estimator"] == "pearson"
        assert am["within_person"][sex]["n"] > 0


@pytest.mark.parametrize(("extra", "recorded"), [([], 10), (["--min-stratum-networks", "3"], 3)])
def test_min_stratum_networks_is_recorded(tmp_path, traits_tsv, extra, recorded):
    """--min-stratum-networks defaults to 10 under --stratify-by and lands in settings; each cell counts small strata."""
    res, path = _run(
        tmp_path,
        traits_tsv,
        "--trait",
        "liability",
        "--stratify-by",
        "depth",
        "--permutations",
        "0",
        "--bootstrap",
        "0",
        *extra,
    )
    assert res.returncode == 0, res.stderr
    am = _load(path)["assortative_mating"]
    assert am["settings"]["min_stratum_networks"] == recorded
    assert {"small_stratum", "degenerate_stratum"} <= set(am["mate_correlation"][0]["n_dropped"])


def test_min_stratum_networks_is_null_without_strata(tmp_path, traits_tsv):
    """Without --stratify-by the minimum has nothing to act on and is recorded as null."""
    res, path = _run(tmp_path, traits_tsv, "--trait", "liability", "--min-stratum-networks", "3", "--bootstrap", "0")
    assert res.returncode == 0, res.stderr
    assert _load(path)["assortative_mating"]["settings"]["min_stratum_networks"] is None


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--trait", "liability", "liability"], "same column twice"),
        (["--trait", "liability", "score", "dx"], "one or two columns"),
        (["--trait", "liability", "--trait-type", "score=continuous"], "not a --trait column"),
        (["--trait", "liability", "--trait-type", "liability=nominal"], "COL=continuous|binary|ordinal"),
        (["--trait", "liability", "--stratify-by", "birth_year"], "needs --birth-year-col"),
        (["--trait", "liability", "--min-stratum-networks", "0"], "expected an integer >= 1"),
        (["--trait", "liability", "--seed", str(2**63)], "argument --seed: expected an integer from -2^63 to 2^63-1"),
        (["--trait", "liability", "--seed", str(-(2**63) - 1)], "argument --seed: expected an integer from -2^63"),
    ],
)
def test_usage_errors_exit_2(tmp_path, traits_tsv, flags, message):
    """Duplicate traits, a --trait-type for a non-trait, birth-year strata without the column, a zero minimum, and a seed outside int64 exit 2."""
    res, path = _run(tmp_path, traits_tsv, *flags)
    assert res.returncode == 2
    assert message in res.stderr
    assert not path.exists()


def test_cli_import_and_parser_leave_numba_unloaded():
    """Importing the CLI and building every subcommand's parser load neither numba, SciPy's optimisers nor the assortative-mating module."""
    code = (
        "import sys; from pedsum.cli import _parse_args; _parse_args(['summarize', '--in', 'x', '--out', 'y']); "
        "print([m for m in ('numba', 'scipy.optimize', 'pedsum.assortative_mating') if m in sys.modules])"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "[]"


def test_seed_takes_the_int64_range(tmp_path, traits_tsv):
    """Both ends of the int64 range run to completion and are recorded as given."""
    for seed in (2**63 - 1, -(2**63)):
        res, path = _run(
            tmp_path, traits_tsv, "--trait", "dx", "--seed", str(seed), "--permutations", "9", "--bootstrap", "9"
        )
        assert res.returncode == 0, res.stderr
        assert _load(path)["assortative_mating"]["settings"]["seed"] == seed


def test_trait_type_column_may_contain_equals():
    """``--trait-type`` splits at the last ``=``, so a column name may contain one."""
    args = _parse_args(
        ["assortative-mating", "--in", "x", "--out", "y", "--trait", "a=b", "--trait-type", "a=b=ordinal"]
    )
    assert args.trait_type == {"a=b": "ordinal"}


def test_small_values_keep_significant_figures(tmp_path):
    """A value below 1e-4 keeps four significant figures in the YAML instead of rounding to 0."""
    n = 25_000
    rng = np.random.default_rng(0)
    ids = np.arange(1, 3 * n + 1)
    pedigree = pl.DataFrame(
        {
            "id": ids,
            "sex": ["F"] * n + ["M"] * n + rng.choice(["F", "M"], n).tolist(),
            "mother": np.r_[np.full(2 * n, -1), ids[:n]],
            "father": np.r_[np.full(2 * n, -1), ids[n : 2 * n]],
            "x": rng.normal(size=3 * n),
        }
    )
    path = tmp_path / "ped.tsv"
    pedigree.write_csv(path, separator="\t")
    res, out = _run(tmp_path, path, "--trait", "x", "--permutations", "0")
    assert res.returncode == 0, res.stderr
    am = _load(out)["assortative_mating"]
    assert am["mating_pairs"]["largest_mate_network_share"] == 4e-05
    assert am["mate_correlation"][0]["largest_mate_network_share"] == 4e-05


@pytest.mark.parametrize(
    ("trait", "message"),
    [
        ("nope", "not in input"),
        ("all_na", "missing in every row"),
        ("const", "constant"),
        ("generation", "--trait-type generation=ordinal"),
        ("infinite", "non-finite"),
        ("bad", "must be numeric"),
    ],
)
def test_data_errors_exit_1(tmp_path, traits_tsv, trait, message):
    """Trait data errors are logged and exit 1 without writing the YAML."""
    res, path = _run(tmp_path, traits_tsv, "--trait", trait)
    assert res.returncode == 1
    assert message in res.stderr
    assert not path.exists()


# ---------------------------------------------------------------------------
# --threads: numba takes the physical cores unless told otherwise
# ---------------------------------------------------------------------------


def _fake_sysfs(tmp_path, siblings: dict[int, str], name: str = "core_cpus_list") -> Path:
    for cpu, text in siblings.items():
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / name).write_text(text + "\n")
    return tmp_path


def test_physical_cores_counts_smt_siblings_once(tmp_path, monkeypatch):
    """Four CPUs in two SMT pairs are two cores; a CPU outside the affinity set does not count."""
    sysfs = _fake_sysfs(tmp_path, {0: "0,2", 1: "1,3", 2: "0,2", 3: "1,3"})
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    assert physical_cores(sysfs) == 2
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 2})
    assert physical_cores(sysfs) == 1
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})
    assert physical_cores(sysfs) == 2


def test_physical_cores_falls_back_without_topology(tmp_path, monkeypatch):
    """Older kernels name the file ``thread_siblings_list`` and ranges parse; a CPU without a topology is a core."""
    sysfs = _fake_sysfs(tmp_path, {0: "0-1", 1: "0-1"}, name="thread_siblings_list")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 5})
    assert physical_cores(sysfs) == 2
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: 7)
    assert physical_cores(sysfs) == 7


def test_threads_default_is_physical_cores_and_explicit_wins(tmp_path, traits_tsv):
    """Without ``--threads`` numba runs on the physical cores (capped by ``NUMBA_NUM_THREADS``); ``--threads N`` sets N; other subcommands keep 1."""
    assert _parse_args(["assortative-mating", "--in", "x", "--out", "y", "--trait", "dx"]).threads is None
    assert (
        _parse_args(["assortative-mating", "--in", "x", "--out", "y", "--trait", "dx", "--threads", "3"]).threads == 3
    )
    assert _parse_args(["summarize", "--in", "x", "--out", "y"]).threads == 1
    env = os.environ | {"NUMBA_NUM_THREADS": "4"}
    code = (
        "import sys; from pedsum.cli import main, physical_cores; sys.argv = ['pedsum', *sys.argv[1:]]; "
        "print(physical_cores()); main()"
    )

    def run(*extra):
        out_dir = tmp_path / f"out{len(extra)}"
        argv = ["assortative-mating", "--in", str(traits_tsv), "--out", str(out_dir), "--trait", "dx", *extra]
        argv += ["--permutations", "9"]
        proc = subprocess.run([sys.executable, "-c", code, *argv], capture_output=True, text=True, env=env, check=False)
        assert proc.returncode == 0, proc.stderr
        return int(proc.stdout.split()[0]), _load(out_dir / "assortative_mating.yaml")["assortative_mating"]

    cores, default = run()
    assert default["settings"]["threads"] == min(cores, 4)
    _, explicit = run("--threads", "2")
    assert explicit["settings"]["threads"] == 2
