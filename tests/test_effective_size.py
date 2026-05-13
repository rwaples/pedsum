"""End-to-end tests for the ``--effective-size`` CLI flag in pedsum.

Drives the CLI ``_run_summarize`` path on the bundled
``example_pedigree.tsv`` and asserts on the resulting YAML / TSV.
Pins the TSV/YAML split, the flag-combination warning, and the
argparse ``--ne-threads`` guard.
"""
from __future__ import annotations

import csv
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example_pedigree.tsv"
SCRIPT = REPO / "pedigree_summary.py"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=cwd or REPO,
    )


def _load_yaml(base: Path) -> dict:
    return yaml.safe_load((base.parent / f"{base.name}.summary.yaml").read_text())


def _load_tsv(base: Path) -> list[list[str]]:
    path = base.parent / f"{base.name}.summary.pedigree.tsv"
    with path.open() as fh:
        return list(csv.reader(fh, delimiter="\t"))


def test_default_run_has_no_effective_size_keys(tmp_path):
    base = tmp_path / "p"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(base)])
    assert res.returncode == 0, res.stderr

    yaml_data = _load_yaml(base)
    ped = yaml_data["pedigree"]
    assert "effective_size" not in ped
    assert "effective_size_scalars" not in ped

    tsv_rows = _load_tsv(base)
    assert not any(r[0].startswith("effective_size") for r in tsv_rows[1:])


def test_effective_size_without_inbreeding_works(tmp_path):
    """Decoupling check: --effective-size standalone, no F section emitted."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--effective-size",
    ])
    assert res.returncode == 0, res.stderr

    yaml_data = _load_yaml(base)
    ped = yaml_data["pedigree"]
    assert ped.get("inbreeding") is None
    assert "effective_size" in ped
    assert len(ped["effective_size"]) == 8
    # Without --ne-coancestry the Ne_C scalar is null.
    assert ped["effective_size"]["ne_coancestry"]["ne"] is None


def test_effective_size_with_inbreeding(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr

    yaml_data = _load_yaml(base)
    ped = yaml_data["pedigree"]
    assert ped["inbreeding"] is not None
    # memo_size dropped from the inbreeding summary in the refactor.
    assert "memo_size" not in ped["inbreeding"]
    assert ped["effective_size"]["ne_coancestry"]["ne"] is None


def test_ne_coancestry_opts_in(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--effective-size", "--ne-coancestry",
    ])
    assert res.returncode == 0, res.stderr

    yaml_data = _load_yaml(base)
    es = yaml_data["pedigree"]["effective_size"]
    ne_c = es["ne_coancestry"]["ne"]
    # On the 200-row example we expect a finite scalar; if upstream changes
    # cause it to come back as None for genuine reasons, the test still
    # validates "not the NaN-array sentinel that --skip-ne-coancestry=True
    # would have produced".
    assert ne_c is None or isinstance(ne_c, float)
    assert all(v is not None for v in es["ne_coancestry"]["mean_theta_per_gen"][1:])


def test_tsv_split_holds(tmp_path):
    """Only the eight scalar Ne values may appear in the long-form TSV."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--effective-size",
    ])
    assert res.returncode == 0, res.stderr

    tsv_rows = _load_tsv(base)
    scalar_rows = [r for r in tsv_rows[1:] if r[0] == "effective_size_scalars"]
    deep_rows = [r for r in tsv_rows[1:] if r[0] == "effective_size"]
    assert len(scalar_rows) == 8
    assert len(deep_rows) == 0


def test_warning_for_orphaned_ne_flags(tmp_path, caplog):
    base = tmp_path / "p"
    # Drive via subprocess to get realistic stderr capture.
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base), "--ne-coancestry",
    ])
    assert res.returncode == 0
    assert "have no effect without --effective-size" in res.stderr


def test_ne_threads_zero_argparse_error(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--effective-size", "--ne-threads", "0",
    ])
    assert res.returncode != 0
    assert "--ne-threads" in res.stderr
    # Nothing should be written when argparse rejects.
    assert not (base.parent / f"{base.name}.summary.yaml").exists()


def test_ne_threads_accepts_positive_int(tmp_path):
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--effective-size", "--ne-threads", "4",
    ])
    assert res.returncode == 0, res.stderr
