"""Tests for the ``--birth-year-col`` flag wiring birth years into Hill Ne.

Without birth years, pedsum's effective-size output for Ne_H collapses to
Ne_V (``collapses_to_ne_v: true`` with a null cohort window). With the
flag, pedigree-graph's overlapping-generation kernel takes over and
populates the cohort window plus the sex-decomposed Ne_m / Ne_f scalars.

The bundled ``example_pedigree.tsv`` lacks a birth-year column, so each
test materialises a small augmented TSV in ``tmp_path`` by deriving
``birth_year = 2000 + 5 * generation``. That keeps every parent-child
edge topologically consistent (child.birth_year > parent.birth_year by
construction) and yields ~5 generations of cohorts for Hill to chew on.
"""
from __future__ import annotations

import csv
from pathlib import Path

from conftest import EXAMPLE
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run


def _write_augmented_pedigree(dest: Path) -> Path:
    """Copy the example pedigree adding a ``birth_year`` column from ``generation``."""
    with EXAMPLE.open() as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    header = rows[0] + ["birth_year"]
    out_rows = [header]
    gen_idx = rows[0].index("generation")
    for row in rows[1:]:
        gen = int(row[gen_idx])
        out_rows.append([*row, str(2000 + 5 * gen)])
    with dest.open("w") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerows(out_rows)
    return dest


def test_hill_collapses_without_birth_year(tmp_path):
    """Baseline: without ``--birth-year-col`` Hill collapses to Ne_V."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--effective-size",
    ])
    assert res.returncode == 0, res.stderr

    hill = _load_yaml(base)["pedigree"]["popgen"]["effective_size"]["ne_hill_overlapping"]
    assert hill["collapses_to_ne_v"] is True
    assert hill["cohort_window"] is None
    assert hill["Ne_m"] is None
    assert hill["Ne_f"] is None
    assert hill["n_eligible_cohorts"] == 0


def test_hill_populated_with_birth_year(tmp_path):
    """``--birth-year-col`` feeds birth years through so Hill uses cohorts."""
    pedigree = _write_augmented_pedigree(tmp_path / "augmented.tsv")
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(pedigree), "--out", str(base),
        "--effective-size", "--birth-year-col", "birth_year",
    ])
    assert res.returncode == 0, res.stderr

    hill = _load_yaml(base)["pedigree"]["popgen"]["effective_size"]["ne_hill_overlapping"]
    assert hill["collapses_to_ne_v"] is False
    assert hill["cohort_window"] is not None
    assert hill["cohort_window"]["c_min"] <= hill["cohort_window"]["c_max"]
    assert hill["n_eligible_cohorts"] >= 1
    # Sex-decomposed scalars must be finite (not the None sentinel).
    assert isinstance(hill["Ne_m"], float)
    assert isinstance(hill["Ne_f"], float)
    # Generation interval should come back as a finite, positive number.
    assert hill["generation_interval"] > 0


def test_birth_year_missing_column_errors(tmp_path):
    """Naming a non-existent column fails with the standard missing-cols error."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--effective-size", "--birth-year-col", "does_not_exist",
    ])
    assert res.returncode == 1
    assert "missing required columns" in res.stderr
    assert "does_not_exist" in res.stderr
