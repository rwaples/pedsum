"""Shared test helpers for pedsum's CLI-driven tests."""

from __future__ import annotations

import contextlib
import csv
import gzip
import io
import logging
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example_pedigree.tsv"
SCRIPT = REPO / "pedigree_summary.py"

# Make the repo-root ``pedigree_summary`` shim importable for every test,
# regardless of collection order or how pytest is invoked. conftest.py is
# imported before any test module, so this replaces the per-file
# ``sys.path.insert`` hacks the direct-import tests used to carry.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pedsum.cli import main  # noqa: E402


def run_pedsum(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the CLI in-process and return a ``CompletedProcess``-shaped result.

    Runs ``pedsum.cli.main`` under redirected stdout/stderr with ``sys.argv``
    and the working directory set as a ``python pedigree_summary.py ...``
    invocation would see them.  A subprocess costs ~1.1s of interpreter and
    import startup per call, which dominated the suite.
    """
    argv = [str(SCRIPT), *args]
    out, err = io.StringIO(), io.StringIO()
    prev_cwd, prev_argv = os.getcwd(), sys.argv
    os.chdir(cwd or REPO)
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(args))
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv
        logging.getLogger().handlers.clear()
    return subprocess.CompletedProcess(argv, code, out.getvalue(), err.getvalue())


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
