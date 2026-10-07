#!/usr/bin/env python3
"""Synthetic pedigree with a known Mate Correlation, for the ``assortative-mating`` cost gate.

Every Mating Pair's (mother t1, mother t2, father t1, father t2) is one draw of
a 4-variate normal with unit variances, within-person correlations
``--within-m`` / ``--within-f`` and the mother x father block ``R_mf``
(``--r-mf``, row-major). Remating keeps that marginal exact: a father with two
mates has one trait vector and each mate is drawn conditional on it, and a
remating mother gets her second father the same way. Children, one per pair,
carry the pair into the pedigree and have no trait values.

Columns: ``id sex mother father birth_year liab dx``. ``liab`` is t1
(continuous); ``dx`` is t2 thresholded at prevalence ``--prevalence`` (binary,
0/1). Parents' birth years are uniform on ``[--year-min, --year-max)``, so
``--stratify-by birth_year --birth-year-bin 10`` sees several strata per sex.

Usage::

    python benchmarks/generate_assortative_mating.py --pairs 100000 --seed 0 --out /tmp/am_1e5.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm


def mate_covariance(r_mf: np.ndarray, within_m: float, within_f: float) -> np.ndarray:
    """The 4x4 covariance of (m1, m2, f1, f2)."""
    mm = np.array([[1.0, within_m], [within_m, 1.0]])
    ff = np.array([[1.0, within_f], [within_f, 1.0]])
    return np.block([[mm, r_mf], [r_mf.T, ff]])


def _conditional(
    rng: np.random.Generator, given: np.ndarray, cov: np.ndarray, known: slice, drawn: slice
) -> np.ndarray:
    """Draws of the ``drawn`` block conditional on the ``known`` block equal to each row of ``given``."""
    s_kk, s_dk, s_dd = cov[known, known], cov[drawn, known], cov[drawn, drawn]
    beta = s_dk @ np.linalg.inv(s_kk)
    resid = s_dd - beta @ s_kk @ beta.T
    noise = rng.multivariate_normal(np.zeros(resid.shape[0]), resid, size=len(given))
    return given @ beta.T + noise


def generate(
    pairs: int,
    seed: int,
    r_mf: np.ndarray,
    within_m: float,
    within_f: float,
    remate_fathers: float,
    remate_mothers: float,
    prevalence: float,
    year_min: int,
    year_max: int,
) -> tuple[pl.DataFrame, dict]:
    """The pedigree frame and a metadata record (counts and the target matrix)."""
    rng = np.random.default_rng(seed)
    cov = mate_covariance(r_mf, within_m, within_f)
    mothers_blk, fathers_blk = slice(0, 2), slice(2, 4)

    n_fathers = max(1, round(pairs / ((1 + remate_fathers) * (1 + remate_mothers))))
    n_remate_f = min(n_fathers, round(remate_fathers * n_fathers))
    n_first = n_fathers + n_remate_f
    n_remate_m = min(n_first, pairs - n_first)

    father_traits = rng.multivariate_normal(np.zeros(2), cov[fathers_blk, fathers_blk], size=n_fathers)
    pair_father = np.concatenate([np.arange(n_fathers), rng.choice(n_fathers, n_remate_f, replace=False)])
    mother_traits = _conditional(rng, father_traits[pair_father], cov, fathers_blk, mothers_blk)
    pair_mother = np.arange(n_first)

    remating_mothers = rng.choice(n_first, n_remate_m, replace=False)
    new_fathers = _conditional(rng, mother_traits[remating_mothers], cov, mothers_blk, fathers_blk)
    father_traits = np.concatenate([father_traits, new_fathers])
    pair_father = np.concatenate([pair_father, n_fathers + np.arange(n_remate_m)])
    pair_mother = np.concatenate([pair_mother, remating_mothers])

    n_mothers, n_fathers_all = len(mother_traits), len(father_traits)
    mother_ids = 1 + np.arange(n_mothers)
    father_ids = 1 + n_mothers + np.arange(n_fathers_all)
    child_ids = 1 + n_mothers + n_fathers_all + np.arange(pairs)
    mother_years = rng.integers(year_min, year_max, n_mothers)
    father_years = rng.integers(year_min, year_max, n_fathers_all)
    child_years = np.maximum(mother_years[pair_mother], father_years[pair_father]) + 25

    traits = np.concatenate([mother_traits, father_traits])
    cut = norm.ppf(1 - prevalence)
    n_parents = n_mothers + n_fathers_all
    df = pl.DataFrame(
        {
            "id": np.concatenate([mother_ids, father_ids, child_ids]),
            "sex": ["F"] * n_mothers + ["M"] * n_fathers_all + ["F", "M"] * (pairs // 2) + ["F"] * (pairs % 2),
            "mother": np.concatenate([np.full(n_parents, -1), mother_ids[pair_mother]]),
            "father": np.concatenate([np.full(n_parents, -1), father_ids[pair_father]]),
            "birth_year": np.concatenate([mother_years, father_years, child_years]),
            "liab": np.concatenate([np.round(traits[:, 0], 6).astype(str), np.full(pairs, "NA")]),
            "dx": np.concatenate([(traits[:, 1] > cut).astype(int).astype(str), np.full(pairs, "NA")]),
        }
    )
    meta = {
        "pairs": pairs,
        "seed": seed,
        "n_mothers": n_mothers,
        "n_fathers": n_fathers_all,
        "fathers_with_two_mates": int(n_remate_f),
        "mothers_with_two_mates": int(n_remate_m),
        "r_mf": r_mf.tolist(),
        "within_m": within_m,
        "within_f": within_f,
        "prevalence": prevalence,
        "rows": len(df),
    }
    return df, meta


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=int, required=True, help="number of Mating Pairs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="output TSV path")
    p.add_argument(
        "--r-mf",
        type=float,
        nargs=4,
        default=[0.30, 0.15, 0.05, 0.25],
        metavar="R",
        help="R_mf row-major (default: %(default)s)",
    )
    p.add_argument("--within-m", type=float, default=0.4)
    p.add_argument("--within-f", type=float, default=0.3)
    p.add_argument(
        "--remate-fathers", type=float, default=0.2, help="share of fathers with two mates (default: %(default)s)"
    )
    p.add_argument(
        "--remate-mothers", type=float, default=0.1, help="share of mothers with two mates (default: %(default)s)"
    )
    p.add_argument("--prevalence", type=float, default=0.1, help="binary trait prevalence (default: %(default)s)")
    p.add_argument("--year-min", type=int, default=1900)
    p.add_argument("--year-max", type=int, default=1980)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the pedigree TSV and its ``.meta.json`` sidecar."""
    args = _parse_args(argv)
    df, meta = generate(
        pairs=args.pairs,
        seed=args.seed,
        r_mf=np.array(args.r_mf).reshape(2, 2),
        within_m=args.within_m,
        within_f=args.within_f,
        remate_fathers=args.remate_fathers,
        remate_mothers=args.remate_mothers,
        prevalence=args.prevalence,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out, separator="\t")
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out} ({meta['rows']:,} rows, {meta['pairs']:,} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
