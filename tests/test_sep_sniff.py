r"""Tests for ``--sep`` delimiter sniffing and explicit choices.

Pedsum's default ``--sep auto`` reads the first non-empty line of the
input, counts ``\t`` / ``,`` / ``;`` / ``|`` occurrences, and falls
back to whitespace (PLINK fam-style) when none of those appear. The
explicit choices ``--sep {tab,comma,semicolon,pipe,whitespace}`` opt
out of sniffing.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from conftest import EXAMPLE
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run
from conftest import write_stripped_pedigree as _strip

if TYPE_CHECKING:
    from pathlib import Path


def _rewrite_with_delim(src_tsv: Path, dest: Path, delim: str) -> Path:
    """Copy ``src_tsv`` (tab-separated) to ``dest`` with a different delimiter."""
    with src_tsv.open() as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    with dest.open("w") as fh:
        csv.writer(fh, delimiter=delim).writerows(rows)
    return dest


def test_auto_sniffs_csv(tmp_path):
    """Default ``--sep auto`` picks comma when the file is comma-separated."""
    pedigree = _rewrite_with_delim(EXAMPLE, tmp_path / "p.csv", ",")
    out = tmp_path / "o"
    res = _run(["summarize", "--in", str(pedigree), "--out", str(out)])
    assert res.returncode == 0, res.stderr
    assert "sniffed comma-separated" in res.stderr
    assert _load_yaml(out)["n_total"] == 200


def test_auto_sniffs_semicolon(tmp_path):
    """Default ``--sep auto`` picks semicolon when that's the delimiter."""
    pedigree = _rewrite_with_delim(EXAMPLE, tmp_path / "p.txt", ";")
    out = tmp_path / "o"
    res = _run(["summarize", "--in", str(pedigree), "--out", str(out)])
    assert res.returncode == 0, res.stderr
    assert "sniffed semicolon-separated" in res.stderr


def test_auto_sniffs_whitespace_plink_style(tmp_path):
    r"""Whitespace-only input (PLINK fam-style) routes through ``r'\s+'``."""
    # Build a minimal fam-shape: id sex mother father, space-separated.
    pedigree = tmp_path / "p.fam"
    with pedigree.open("w") as fh:
        fh.write("id sex mother father\n")
        fh.write("1 F -1 -1\n")
        fh.write("2 M -1 -1\n")
        fh.write("3 F 1 2\n")
        fh.write("4 M 1 2\n")
    out = tmp_path / "o"
    res = _run(
        [
            "summarize",
            "--in",
            str(pedigree),
            "--out",
            str(out),
            "--no-effective-size",
            "--no-inbreeding",
        ]
    )
    assert res.returncode == 0, res.stderr
    assert "sniffed whitespace-separated" in res.stderr


def test_explicit_tab_still_works(tmp_path):
    """``--sep tab`` produces the same result as auto-sniff on a TSV file."""
    out = tmp_path / "o"
    res = _run(
        ["summarize", "--in", str(EXAMPLE), "--out", str(out), "--sep", "tab"]
    )
    assert res.returncode == 0, res.stderr
    # No sniff log when sep is explicit.
    assert "sniffed" not in res.stderr


def test_explicit_comma_required_when_no_sniff(tmp_path):
    """``--sep comma`` reads a comma-separated file without sniffing."""
    pedigree = _rewrite_with_delim(EXAMPLE, tmp_path / "p.csv", ",")
    out = tmp_path / "o"
    res = _run(
        ["summarize", "--in", str(pedigree), "--out", str(out), "--sep", "comma"]
    )
    assert res.returncode == 0, res.stderr
    assert "sniffed" not in res.stderr


def test_explicit_tab_on_csv_raises_friendly_error(tmp_path):
    """``--sep tab`` on a CSV input keeps the friendly CSV detection error."""
    pedigree = _strip(tmp_path / "tsv_to_be_made_csv.tsv")
    csv_path = _rewrite_with_delim(pedigree, tmp_path / "p.csv", ",")
    out = tmp_path / "o"
    res = _run(
        ["summarize", "--in", str(csv_path), "--out", str(out), "--sep", "tab"]
    )
    assert res.returncode == 1
    assert "appears to be CSV" in res.stderr


def test_validate_subcommand_also_sniffs(tmp_path):
    """The ``validate`` subcommand picks up ``--sep`` the same way."""
    pedigree = _rewrite_with_delim(EXAMPLE, tmp_path / "p.csv", ",")
    out = tmp_path / "o"
    res = _run(["validate", "--in", str(pedigree), "--out", str(out)])
    assert res.returncode == 0, res.stderr
    assert "sniffed comma-separated" in res.stderr
