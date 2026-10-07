#!/usr/bin/env python3
"""Cross-check ``pedsum assortative-mating`` against simACE on simACE's own pedigrees.

Manual, not run in CI. For each simACE replicate it exports ``pedigree.parquet``
to a pedsum TSV and runs ``pedigree_summary.py assortative-mating --trait
liability1 liability2`` with permutations off, so each cell gets its crude
Pearson r and sandwich SE. Then every check in ``CHECKS`` runs on that
replicate:

- ``continuous R_mf``: each cell must match simACE's ``compute_mate_correlation``
  (run on the same parquet in simACE's pixi env) to ``TOLERANCE`` with the same
  pair count.
- ``tetrachoric``: each liability is cut at the ``1 - K`` quantile of its values
  over the pedigree (affected = above it) for each ``K`` in ``PREVALENCES``, and
  pedsum runs again on the two binary traits with its default inference. Each
  cell's tetrachoric ρ̂ is compared with pedsum's continuous r of the same cell,
  the correlation of the liabilities the tetrachoric should recover on the same
  pairs. The comparison is reported with its z-score and whether r falls inside
  the tetrachoric CI. The pair count is asserted everywhere. A loose z bound
  (``TETRACHORIC_Z_LIMIT``) is asserted only where simACE draws mates bivariate
  normal (``gaussian_mates``); its moment-matched pairing of two assorting
  traits makes no such promise, so there the gap is reported, not asserted.
  Phi is reported beside the phi that a bivariate normal with correlation r and
  the cell's prevalences implies, to show the attenuation.

The configured assortative-mating targets are printed beside pedsum's
continuous estimates for information only: simACE targets a slot-level Pearson
weighted by mating count, so neither side is expected to hit them exactly.

pedsum runs in this process through ``pedsum.cli.main``, the console entry
point, because ``_write_yaml`` rounds every float (to 4 dp, or 4 significant
digits below 0.1). The payload is captured on its way into ``_write_yaml``, and
the YAML written to disk must equal it after that rounding, so the
full-precision values are the ones the CLI reported.

Eligible pairs. simACE keeps each distinct (mother, father) of a child whose
parent ids are both not ``-1``, then drops pairs with a parent absent from the
frame. pedsum's Mating Pairs are the distinct (mother, father) of children
whose parents are both known, and each cell keeps the pairs with both values
present. pedsum refuses a pedigree that names a parent absent from it (the
``parent_refs_present_*`` validation checks), so on any input it accepts
simACE's absent-parent drop removes nothing. With every liability finite
(checked on export) the two pair sets coincide, which the per-cell ``n``
comparison confirms. Thresholding keeps every value, so the tetrachoric cells
use the same pairs as the continuous ones.

Run from the repo root::

    pixi run python benchmarks/validate_assortative_mating.py \
        --simace-root /data/Documents/simACE --work /tmp/am_check

Results go to ``benchmarks/results/assortative_mating_simace.md``. Exit status
is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import polars as pl
import yaml
from scipy.special import ndtri

# Make ``pedsum`` importable when run as ``python benchmarks/...`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsum import cli as pedsum_cli
from pedsum.assortative_mating import bvn_cdf
from pedsum.report import ASSORTATIVE_MATING_FIGURES, _round_floats

if TYPE_CHECKING:
    from collections.abc import Callable

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results" / "assortative_mating_simace.md"
TOLERANCE = 1e-10
TRAITS = ("liability1", "liability2")
#: (mother trait index, father trait index); simACE's ``matrix[i][j]`` is mother trait i x father trait j.
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
#: Prevalences K the liabilities are cut at for the tetrachoric check.
PREVALENCES = (0.1, 0.3)
#: Largest accepted |z| for tetrachoric minus continuous, z = diff / √(se_tet² − se_r²), on replicates whose
#: mates are bivariate normal by construction. Under bivariate normality Pearson r is the efficient estimator
#: of the same ρ and the tetrachoric a consistent one, so their difference has variance se_tet² − se_r² to
#: first order. Over the 32 asserted comparisons (4 replicates x 2 K x 4 cells) a bound of 4 fails by chance
#: with probability about 32 x 6.3e-5 = 0.2%, and it still catches a bias of several SEs, such as a threshold
#: or table error. It is loose because liabilities pooled over generations are only approximately normal.
TETRACHORIC_Z_LIMIT = 4.0
INFERENCE_OFF = ("--permutations", "0", "--bootstrap", "0")


@dataclass(frozen=True)
class Rep:
    """One simACE replicate: ``results/{folder}/{scenario}/rep{rep}``."""

    folder: str
    scenario: str
    rep: int

    def parquet(self, simace_root: Path) -> Path:
        """The replicate's ``pedigree.parquet``."""
        return simace_root / "results" / self.folder / self.scenario / f"rep{self.rep}" / "pedigree.parquet"

    def __str__(self) -> str:
        """``folder/scenario/repN``."""
        return f"{self.folder}/{self.scenario}/rep{self.rep}"


DEFAULT_REPS = (
    Rep("epimight_h2_bias", "am", 1),
    Rep("epimight_h2_bias", "am_asym", 1),
    Rep("epimight_h2_bias", "am_t1", 1),
    Rep("examples", "am_none", 1),
    Rep("examples", "am_none", 2),
    Rep("examples", "am_none", 3),
)

# Runs in simACE's env: argv = parquet, scenario, config dir. Prints one JSON object.
SIMACE_SNIPPET = """
import json, sys
import numpy as np
import polars as pl
from simace.analysis.stats.correlations import compute_mate_correlation
from simace.config import resolve_defaults, resolve_scenarios
from simace.plotting.plot_utils import param_as_float
from simace.simulation.mate_correlation import expected_mate_corr_matrix

parquet, scenario, config_dir = sys.argv[1:4]
df = pl.read_parquet(parquet)
mate = compute_mate_correlation(df)

mothers, fathers = df["mother"].to_numpy(), df["father"].to_numpy()
both = (mothers != -1) & (fathers != -1)
base = np.int64(max(int(mothers.max()), int(fathers.max())) + 1)
n_any = len(np.unique(mothers[both].astype(np.int64) * base + fathers[both]))
n_half = int(((mothers == -1) != (fathers == -1)).sum())

defaults = resolve_defaults(config_dir)
p = defaults | resolve_scenarios(config_dir, defaults)[scenario]
target = expected_mate_corr_matrix(
    assort1=param_as_float(p.get("assort1", 0)),
    assort2=param_as_float(p.get("assort2", 0)),
    rA=float(p.get("rA", 0)),
    rC=float(p.get("rC", 0)),
    A1=float(p.get("A1", 0)),
    C1=param_as_float(p.get("C1", 0)),
    A2=float(p.get("A2", 0)),
    C2=param_as_float(p.get("C2", 0)),
    assort_matrix=p.get("assort_matrix"),
    rE=float(p.get("rE", 0)),
    E1=param_as_float(p.get("E1", 0)),
    E2=param_as_float(p.get("E2", 0)),
)
print(json.dumps({
    "rows": df.height,
    "matrix": mate["matrix"],
    "n_pairs": mate["n_pairs"],
    "n_pairs_any_parent": n_any,
    "n_half_founders": n_half,
    "assort": {k: p.get(k) for k in ("assort1", "assort2", "assort_matrix")},
    "moment_matched": bool(param_as_float(p.get("assort1", 0)) and param_as_float(p.get("assort2", 0))),
    "target": target.tolist(),
}))
"""


def write_pedsum_tsv(parquet: Path, out: Path, traits: dict[str, pl.Expr]) -> None:
    """Export ``parquet`` as a pedsum TSV with ``traits`` as trait columns.

    simACE and pedsum share both codings: sex 0=female, 1=male
    (``simace/core/relationships.py:41``; ``pedsum/base.py``, passed as
    ``--sex-encoding default`` so auto-detection cannot pick PLINK), and ``-1``
    for an unknown parent (pedsum keeps id ``0`` as a real parent because the
    command never sets ``zero_as_missing``). So the columns pass through as
    they are; the guards below make that a checked fact.
    Floats are written at shortest round-trip precision and read back to prove
    pedsum sees the parquet's exact values.
    """
    out_df = (
        pl.scan_parquet(parquet)
        .select("id", "sex", "mother", "father", *(expr.alias(name) for name, expr in traits.items()))
        .collect()
    )
    bad_sex = out_df.filter(~pl.col("sex").is_in([0, 1])).height
    if bad_sex:
        raise SystemExit(f"{parquet}: {bad_sex} rows with sex outside {{0, 1}}")
    for name in traits:
        if not out_df[name].is_finite().all():
            raise SystemExit(f"{parquet}: trait {name!r} has non-finite values; the eligible pair sets would differ")
    out_df.write_csv(out, separator="\t")
    back = pl.read_csv(out, separator="\t", columns=list(traits), schema_overrides=dict.fromkeys(traits, pl.Float64))
    for name in traits:
        if not np.array_equal(back[name].to_numpy(), out_df[name].cast(pl.Float64).to_numpy()):
            raise SystemExit(f"{out}: trait {name!r} does not round-trip through the TSV")


def run_pedsum(tsv: Path, out_dir: Path, traits: tuple[str, ...], *inference: str) -> dict:
    """``assortative-mating`` on ``tsv`` with the ``inference`` flags; return the unrounded ``assortative_mating`` block."""
    argv = [
        "assortative-mating",
        "--in",
        str(tsv),
        "--out",
        str(out_dir),
        "--trait",
        *traits,
        "--sex-encoding",
        "default",
        *inference,
        "-q",
    ]
    written: list[dict] = []
    write_yaml = pedsum_cli._write_yaml

    def capture(data: dict, path: Path, figures: int = 0) -> None:
        written.append(data)
        write_yaml(data, path, figures)

    with mock.patch.object(pedsum_cli, "_write_yaml", capture):
        code = pedsum_cli.main(argv)
    if code != 0:
        raise SystemExit(f"pedsum assortative-mating exited {code} on {tsv}")
    (payload,) = written
    on_disk = yaml.safe_load((out_dir / "assortative_mating.yaml").read_text())
    if on_disk != _round_floats(payload, figures=ASSORTATIVE_MATING_FIGURES):
        raise SystemExit(f"{out_dir}: assortative_mating.yaml differs from the captured payload after rounding")
    return payload["assortative_mating"]


def pedsum_on(parquet: Path, work: Path, name: str, traits: dict[str, pl.Expr], *inference: str) -> dict:
    """Export ``traits`` of ``parquet`` to ``work/{name}.tsv``, run pedsum on it, and drop the TSV."""
    tsv = work / f"{name}.tsv"
    write_pedsum_tsv(parquet, tsv, traits)
    try:
        return run_pedsum(tsv, work / name, tuple(traits), *inference)
    finally:
        tsv.unlink()


def run_simace(simace_root: Path, rep: Rep) -> dict:
    """Run simACE's ``compute_mate_correlation`` and resolve the configured target for ``rep`` in simACE's env."""
    cmd = [
        "pixi",
        "run",
        "--manifest-path",
        str(simace_root / "pixi.toml"),
        "python",
        "-c",
        SIMACE_SNIPPET,
        str(rep.parquet(simace_root)),
        rep.scenario,
        str(simace_root / "config"),
    ]
    done = subprocess.run(cmd, check=True, cwd=simace_root, capture_output=True, text=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


@dataclass(frozen=True)
class RepContext:
    """What a check sees for one replicate: simACE's output and pedsum's continuous run on the liabilities."""

    rep: Rep
    parquet: Path
    work: Path
    simace: dict
    liability: dict


@dataclass(frozen=True)
class CheckResult:
    """One check on one replicate: a formatted row per compared cell, the check's metric, and the verdict."""

    rows: list[tuple[str, ...]]
    worst: float
    passed: bool
    note: str = ""


@dataclass(frozen=True)
class Check:
    """A named comparison, its explanation and table columns, what its metric is, and how to run it."""

    name: str
    about: str
    columns: tuple[str, ...]
    metric: str
    run: Callable[[RepContext], CheckResult]


def _cells(block: dict) -> dict[tuple[str, str], dict]:
    """The ``mate_correlation`` records of a pedsum block by (mother trait, father trait)."""
    return {(c["mother"], c["father"]): c for c in block["mate_correlation"]}


def _cell_label(cell: tuple[int, int]) -> str:
    i, j = cell
    return f"mother {TRAITS[i]} x father {TRAITS[j]}"


def _worst(values: list[float]) -> float:
    """The largest value, with NaN counted as infinite."""
    return max(np.inf if np.isnan(v) else v for v in values)


def continuous_rmf(ctx: RepContext) -> CheckResult:
    """Continuous ``R_mf``: pedsum's crude Pearson per cell against simACE's ``compute_mate_correlation``."""
    by_cell = _cells(ctx.liability)
    rows, diffs, same_n = [], [], True
    for i, j in CELLS:
        cell = by_cell[TRAITS[i], TRAITS[j]]
        r_pedsum = cell["crude"]["pearson"]["r"]
        r_simace = ctx.simace["matrix"][i][j]
        diff = abs(r_pedsum - r_simace)
        diffs.append(diff)
        same_n &= cell["n"] == ctx.simace["n_pairs"]
        rows.append(
            (
                _cell_label((i, j)),
                str(cell["n"]),
                str(ctx.simace["n_pairs"]),
                f"{r_pedsum:.17g}",
                f"{r_simace:.17g}",
                f"{diff:.3g}",
            )
        )
    worst = _worst(diffs)
    return CheckResult(rows, worst, same_n and worst <= TOLERANCE)


def implied_phi(table: list[list[int]], r: float) -> float:
    """Phi of a bivariate normal with correlation ``r`` cut at the 2x2 ``table``'s own margins (level 1 = affected)."""
    (a, b), (c, d) = table
    n = a + b + c + d
    p_m, p_f = (c + d) / n, (b + d) / n
    both = float(bvn_cdf(np.array([-ndtri(1 - p_m)]), np.array([-ndtri(1 - p_f)]), r)[0])
    return (both - p_m * p_f) / math.sqrt(p_m * (1 - p_m) * p_f * (1 - p_f))


def gaussian_mates(simace: dict) -> bool:
    """Whether simACE drew this replicate's mate liabilities bivariate normal.

    Random mating and single-trait assortment (a Gaussian copula on the
    assorting trait's ranks) do; with both traits assorting,
    ``_assortative_pair_partners`` (``simace/simulation/simulate.py``) matches
    the four mate correlations by sorting and swaps, which "is not a copula and
    draws no multivariate normal".
    """
    return not simace["moment_matched"]


def tetrachoric_vs_continuous(ctx: RepContext) -> CheckResult:
    """Tetrachoric ρ̂ of the liabilities cut at each prevalence against pedsum's continuous r on the same pairs."""
    continuous = _cells(ctx.liability)
    rows, zs, same_n, n_in_ci = [], [], True, 0
    for k in PREVALENCES:
        names = tuple(f"{t}_top{round(100 * k)}" for t in TRAITS)
        cut = {
            name: (pl.col(t) > pl.col(t).quantile(1 - k)).cast(pl.Int8) for name, t in zip(names, TRAITS, strict=True)
        }
        binary = _cells(pedsum_on(ctx.parquet, ctx.work, f"top{round(100 * k)}", cut))
        for i, j in CELLS:
            cell, reference = binary[names[i], names[j]], continuous[TRAITS[i], TRAITS[j]]
            tet, pearson = cell["crude"]["tetrachoric"], reference["crude"]["pearson"]
            same_n &= cell["n"] == reference["n"]
            rho, r, ci = tet["rho"], pearson["r"], tet["ci"]
            if rho is None:
                zs.append(math.nan)
                rows.append((f"{k:g}", _cell_label((i, j)), str(cell["n"]), f"null ({tet['reason']})", *["-"] * 8))
                continue
            diff = rho - r
            excess = (tet["se"] or math.nan) ** 2 - (pearson["se"] or math.nan) ** 2
            z = diff / math.sqrt(excess) if excess > 0 else math.nan
            zs.append(abs(z))
            inside = ci is not None and ci[0] <= r <= ci[1]
            n_in_ci += inside
            draws = tet["permutations"]["draws_used"]
            rows.append(
                (
                    f"{k:g}",
                    _cell_label((i, j)),
                    str(cell["n"]),
                    f"{rho:.4f}",
                    "null" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                    f"{r:.4f}",
                    f"{diff:+.4f}",
                    f"{z:+.2f}",
                    "yes" if inside else "no",
                    f"{cell['crude']['phi']['r']:.4f}",
                    f"{implied_phi(cell['crude']['table'], r):.4f}",
                    f"{tet['p_perm']:.3g} ({draws})",
                )
            )
    worst = _worst(zs)
    note = f"r inside the tetrachoric CI in {n_in_ci}/{len(zs)} cells"
    if not gaussian_mates(ctx.simace):
        return CheckResult(rows, worst, same_n, f"{note}; z not asserted (moment-matched mates)")
    return CheckResult(rows, worst, same_n and worst <= TETRACHORIC_Z_LIMIT, note)


CONTINUOUS_ABOUT = (
    f"pedsum's crude Pearson r of each cell (`--permutations 0 --bootstrap 0`) against simACE's "
    f"`compute_mate_correlation` on the same parquet. Passes when every cell has simACE's pair count and "
    f"|pedsum - simACE| <= {TOLERANCE:g}."
)

TETRACHORIC_ABOUT = (
    "Each liability is cut at the 1 - K quantile of its values over the whole pedigree (affected = above it), "
    "and pedsum runs on the two binary traits with its default inference (sandwich CI, 999 permutations with "
    "sequential stopping). The tetrachoric ρ̂ of each cell is compared with pedsum's continuous r of the same "
    "cell, the correlation of the liabilities the tetrachoric should recover on the same pairs. `z` is "
    "(tetrachoric - r) / √(se_tet² - se_r²): under bivariate normality r is the efficient estimator of the same "
    "ρ, so that is the SD of the difference to first order. `r in CI` says whether r falls inside the "
    "tetrachoric's 95% sandwich CI. The pair count is asserted on every replicate, and |z| <= "
    f"{TETRACHORIC_Z_LIMIT:g} only where simACE draws mates bivariate normal: random mating (`am_none`) and "
    "single-trait assortment (`am_t1`, a Gaussian copula on the assorting trait's ranks). With both traits "
    "assorting (`am`, `am_asym`) simACE matches the four mate correlations by sorting and swaps, which its "
    "docstring says is not a copula and draws no multivariate normal; there the gap is reported only. "
    "`phi expected` is the phi of a bivariate normal "
    "with correlation r cut at the cell's own pair-weighted prevalences: phi falls well below the latent "
    "correlation at low prevalence, and the tetrachoric undoes that. `p_perm (draws)` is the tetrachoric's "
    "permutation p-value and the permutations it read before stopping."
)

#: Each check compares one statistic per cell on one replicate.
CHECKS: tuple[Check, ...] = (
    Check(
        "continuous R_mf",
        CONTINUOUS_ABOUT,
        ("cell", "n pedsum", "n simACE", "pedsum", "simACE", "abs diff"),
        "max abs diff",
        continuous_rmf,
    ),
    Check(
        "tetrachoric vs continuous R_mf",
        TETRACHORIC_ABOUT,
        (
            "K",
            "cell",
            "n",
            "tetrachoric",
            "95% CI",
            "continuous r",
            "diff",
            "z",
            "r in CI",
            "phi",
            "phi expected",
            "p_perm (draws)",
        ),
        "max abs z",
        tetrachoric_vs_continuous,
    ),
)


def _table(columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    return [
        f"| {' | '.join(columns)} |",
        f"|{'|'.join('---' for _ in columns)}|",
        *(f"| {' | '.join(row)} |" for row in rows),
    ]


def render(simace_root: Path, results: list[tuple[RepContext, list[CheckResult]]], header: dict) -> str:
    """The results file: provenance, pair sets, one section per check, and the target gap."""
    lines = [
        "# pedsum assortative-mating vs simACE",
        "",
        f"Generated by `benchmarks/validate_assortative_mating.py` on {header['date']}.",
        f"pedsum `{header['pedsum_rev']}` (worktree; uncommitted changes: {header['pedsum_dirty']}), "
        f"simACE `{header['simace_rev']}` at `{simace_root}`.",
        "Inputs are each replicate's `pedigree.parquet`; pedsum ran with `--sex-encoding default` and its default "
        f"`--threads` ({header['threads']} numba threads).",
        "",
        "## Pair sets",
        "",
        "simACE (`compute_mate_correlation`) counts each distinct (mother, father) of a child with both parent "
        "ids set, then drops pairs with a parent not in the frame. pedsum counts each distinct Mating Pair of "
        "a child with both parents known, and each cell keeps the pairs with both values present. pedsum "
        "refuses a pedigree that names an absent parent (`parent_refs_present_*`), so simACE's drop removes "
        "nothing on any input pedsum accepts. `n_total` is pedsum's Mating Pair count; `any-parent` is "
        "simACE's distinct pair count before its absent-parent drop, `n_pairs` after it.",
        "",
        "| replicate | rows | pedsum n_total | simACE any-parent | simACE n_pairs | half-founders |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ctx, _ in results:
        meta = ctx.liability["mating_pairs"]
        lines.append(
            f"| {ctx.rep} | {ctx.simace['rows']} | {meta['n_total']} | {ctx.simace['n_pairs_any_parent']} "
            f"| {ctx.simace['n_pairs']} | {ctx.simace['n_half_founders']} |"
        )
    for k, check in enumerate(CHECKS):
        rows = [(str(ctx.rep), *row) for ctx, checks in results for row in checks[k].rows]
        summary = [
            (str(ctx.rep), f"{checks[k].worst:.3g}", checks[k].note or "-", "yes" if checks[k].passed else "NO")
            for ctx, checks in results
        ]
        lines += ["", f"## {check.name}", "", check.about, ""]
        lines += _table(("replicate", *check.columns), rows)
        lines += ["", *_table(("replicate", check.metric, "note", "passed"), summary)]
    lines += [
        "",
        "## Configured targets (information only, not asserted)",
        "",
        "Target is `expected_mate_corr_matrix` on the scenario's resolved config "
        "(`config/_default.yaml` plus the scenario file), as simACE's mate-correlation plot computes it. "
        "simACE aims at a slot-level Pearson weighted by mating count and may not converge "
        "(`simace/simulation/simulate.py`); pedsum measures over distinct Mating Pairs.",
        "",
        "| replicate | assort1 | assort2 | assort_matrix | cell | target | pedsum | pedsum - target |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for ctx, _ in results:
        assort = ctx.simace["assort"]
        by_cell = _cells(ctx.liability)
        for i, j in CELLS:
            target = ctx.simace["target"][i][j]
            r = by_cell[TRAITS[i], TRAITS[j]]["crude"]["pearson"]["r"]
            lines.append(
                f"| {ctx.rep} | {assort['assort1']} | {assort['assort2']} | {assort['assort_matrix']} "
                f"| {_cell_label((i, j))} | {target:.4f} | {r:.4f} | {r - target:+.4f} |"
            )
    return "\n".join(lines) + "\n"


def _git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    """Run every check on every replicate, write the results file, and return 1 on any failure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--simace-root", type=Path, default=REPO.parents[1], help="simACE checkout (default: %(default)s)"
    )
    parser.add_argument("--work", type=Path, default=None, help="scratch dir for the TSVs (default: a temp dir)")
    parser.add_argument(
        "--rep",
        nargs=3,
        action="append",
        metavar=("FOLDER", "SCENARIO", "REP"),
        help="replicate to check; repeatable (default: am, am_asym, am_t1 rep1 and am_none rep1-3)",
    )
    args = parser.parse_args()
    reps = [Rep(f, s, int(r)) for f, s, r in args.rep] if args.rep else list(DEFAULT_REPS)
    with tempfile.TemporaryDirectory(dir=args.work) as tmp:
        results = []
        for rep in reps:
            parquet = rep.parquet(args.simace_root)
            t0 = time.perf_counter()
            simace = run_simace(args.simace_root, rep)
            work = Path(tmp) / str(rep).replace("/", "_")
            work.mkdir()
            liability = pedsum_on(parquet, work, "liability", {t: pl.col(t) for t in TRAITS}, *INFERENCE_OFF)
            ctx = RepContext(rep, parquet, work, simace, liability)
            checks = []
            for check in CHECKS:
                result = check.run(ctx)
                checks.append(result)
                status = "ok" if result.passed else "FAIL"
                note = f"; {result.note}" if result.note else ""
                print(
                    f"{rep}: {check.name} {check.metric} {result.worst:.3g} [{status}]{note} "
                    f"({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
            results.append((ctx, checks))
    header = {
        "date": time.strftime("%Y-%m-%d"),
        "pedsum_rev": _git(REPO, "rev-parse", "--short", "HEAD"),
        "pedsum_dirty": "yes" if _git(REPO, "status", "--porcelain") else "no",
        "simace_rev": _git(args.simace_root, "describe", "--always", "--dirty"),
        "threads": results[0][0].liability["settings"]["threads"],
    }
    text = render(args.simace_root, results, header)
    RESULTS.write_text(text)
    print(text)
    return 0 if all(c.passed for _, checks in results for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
