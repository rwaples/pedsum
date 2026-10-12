"""Shared test helpers for pedsum's CLI-driven tests."""

from __future__ import annotations

import contextlib
import csv
import gzip
import io
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from pedsum.cli import main
from pedsum.pedigree_ops import IdIndex, rows_of_parents
from pedsum.validate import ValidationConfig, ValidationContext, run_checks

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example_pedigree.tsv"

if TYPE_CHECKING:
    import numpy as np

    from pedsum.checks import CheckResult


def run_pedsum(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the CLI in-process and return a ``CompletedProcess``-shaped result.

    Runs ``pedsum.cli.main`` under redirected stdout/stderr with the working
    directory set as a ``pedsum ...`` invocation would see it.  A subprocess
    costs ~1.1s of interpreter and import startup per call, which dominated
    the suite.
    """
    argv = ["pedsum", *args]
    out, err = io.StringIO(), io.StringIO()
    prev_cwd = os.getcwd()
    os.chdir(cwd or REPO)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(args))
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        os.chdir(prev_cwd)
        logging.getLogger().handlers.clear()
    return subprocess.CompletedProcess(argv, code, out.getvalue(), err.getvalue())


def run_check(name: str, rows: list[dict], config: ValidationConfig | None = None) -> CheckResult:
    """Check ``name``'s result when the whole registry runs over ``rows`` in validate's accumulating mode."""
    df = pl.DataFrame(rows, schema=dict.fromkeys(rows[0], pl.String))
    results, _ = run_checks(ValidationContext(df, config or ValidationConfig()))
    return results[name]


def write_ped(path, rows):
    """Write a list of ``{id, sex, mother, father, ...}`` dicts to a TSV."""
    pl.DataFrame(rows).write_csv(path, separator="\t")
    return path


def write_stripped_pedigree(dest: Path) -> Path:
    """Write ``example_pedigree.tsv`` to ``dest`` with ``birth_year`` dropped."""
    with EXAMPLE.open() as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    by_idx = rows[0].index("birth_year")
    keep = [[c for i, c in enumerate(row) if i != by_idx] for row in rows]
    with dest.open("w") as fh:
        csv.writer(fh, delimiter="\t").writerows(keep)
    return dest


def load_summary_yaml(out_dir: Path) -> dict:
    """Parse ``summary.yaml`` produced by a ``summarize`` run."""
    import yaml

    return yaml.safe_load((out_dir / "summary.yaml").read_text())


def load_summary_extra_yaml(out_dir: Path) -> dict:
    """Parse ``summary.extra.yaml`` produced by a ``summarize`` run."""
    import yaml

    return yaml.safe_load((out_dir / "summary.extra.yaml").read_text())


def load_summary_tsv(out_dir: Path) -> list[list[str]]:
    """Read the long-form ``summary.pedigree.tsv`` rows."""
    path = out_dir / "summary.pedigree.tsv"
    with path.open() as fh:
        return list(csv.reader(fh, delimiter="\t"))


def parent_rows(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Each row's (mother row, father row), ``-1`` when unknown, as summarize reads them off the graph."""
    id_index = IdIndex(df["id"].to_numpy())
    mother_rows, _ = rows_of_parents(df["mother"].to_numpy(), id_index)
    father_rows, _ = rows_of_parents(df["father"].to_numpy(), id_index)
    return mother_rows, father_rows


def read_tsv_gz(path: Path, *, as_str: bool = False) -> pl.DataFrame:
    """Read a gzipped TSV into a polars frame (all-string when ``as_str``)."""
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    if as_str:
        return pl.read_csv(data, separator="\t", infer_schema=False)
    return pl.read_csv(data, separator="\t")


def load_validate_tsv_gz(out_dir: Path) -> pl.DataFrame:
    """Read ``validate.tsv.gz`` as strings (matches 0.8 sex encoding)."""
    return read_tsv_gz(out_dir / "validate.tsv.gz", as_str=True)


def load_annotated_tsv_gz(out_dir: Path) -> pl.DataFrame:
    """Read ``annotated.tsv.gz`` with default dtype inference."""
    return read_tsv_gz(out_dir / "annotated.tsv.gz")
