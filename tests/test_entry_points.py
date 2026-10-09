"""The installed ``pedsum`` command and ``python -m pedsum`` (pedsum#18).

Both run in a subprocess, so the exit code is the one a shell sees: ``main``'s
return value, which is 1 when ``validate`` finds a problem.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# The console script sits beside the interpreter in the env that installed pedsum.
PEDSUM = Path(sys.executable).parent / "pedsum"


@pytest.fixture
def ped_with_missing_parent(tmp_path: Path) -> Path:
    """Individual 3's mother is not in the pedigree: a validate finding."""
    ped = tmp_path / "ped.tsv"
    ped.write_text("id\tsex\tmother\tfather\n1\tF\t-1\t-1\n2\tM\t-1\t-1\n3\tM\t9\t2\n")
    return ped


@pytest.mark.parametrize("entry", [[str(PEDSUM)], [sys.executable, "-m", "pedsum"]], ids=["pedsum", "python-m"])
def test_entry_point_returns_main_exit_code(entry: list[str], ped_with_missing_parent: Path, tmp_path: Path) -> None:
    """A validate finding exits 1, whichever way pedsum is started."""
    out = tmp_path / "out"
    proc = subprocess.run(
        [*entry, "validate", "--in", str(ped_with_missing_parent), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1, proc.stderr
    assert "pedsum: validating" in proc.stdout + proc.stderr


def test_recorded_command_starts_with_pedsum(tmp_path: Path) -> None:
    """``summary.yaml`` records ``pedsum <args>``, not the path of the script that ran."""
    ped = tmp_path / "ped.tsv"
    ped.write_text("id\tsex\tmother\tfather\n1\tF\t-1\t-1\n2\tM\t-1\t-1\n3\tM\t1\t2\n")
    args = ["summarize", "--in", str(ped), "--out", str(tmp_path / "out")]
    proc = subprocess.run([sys.executable, "-m", "pedsum", *args], capture_output=True, text=True, check=False)

    assert proc.returncode == 0, proc.stderr
    summary = yaml.safe_load((tmp_path / "out" / "summary.yaml").read_text())
    assert summary["command"] == shlex.join(["pedsum", *args])


@pytest.mark.parametrize("entry", [[str(PEDSUM)], [sys.executable, "-m", "pedsum"]], ids=["pedsum", "python-m"])
def test_entry_point_help_names_pedsum(entry: list[str]) -> None:
    """``--help`` exits 0 and shows ``pedsum`` as the program name."""
    proc = subprocess.run([*entry, "--help"], capture_output=True, text=True, check=False)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("usage: pedsum ")
