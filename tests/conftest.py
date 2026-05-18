"""Shared test helpers for pedsum's ``summarize`` subprocess-driven tests."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example_pedigree.tsv"
SCRIPT = REPO / "pedigree_summary.py"


def run_pedsum(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke ``pedigree_summary.py`` as a subprocess; capture stdout/stderr."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=cwd or REPO,
    )


def write_ped(path, rows):
    """Write a list of ``{id, sex, mother, father, ...}`` dicts to a TSV."""
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def load_summary_yaml(base: Path) -> dict:
    """Parse ``BASENAME.summary.yaml`` produced by a ``summarize`` run."""
    import yaml

    return yaml.safe_load((base.parent / f"{base.name}.summary.yaml").read_text())


def load_summary_extra_yaml(base: Path) -> dict:
    """Parse ``BASENAME.summary.extra.yaml`` produced by a ``summarize`` run."""
    import yaml

    return yaml.safe_load((base.parent / f"{base.name}.summary.extra.yaml").read_text())


def load_summary_tsv(base: Path) -> list[list[str]]:
    """Read the long-form ``BASENAME.summary.pedigree.tsv`` rows."""
    path = base.parent / f"{base.name}.summary.pedigree.tsv"
    with path.open() as fh:
        return list(csv.reader(fh, delimiter="\t"))
