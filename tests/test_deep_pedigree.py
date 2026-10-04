"""Deep-pedigree guards in ``summarize``: the depth-aware F warning and the descendant-path overflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from conftest import EXAMPLE, run_pedsum, write_ped

import pedsum.cli

if TYPE_CHECKING:
    from pathlib import Path


def _sib_mating_chain(path: Path, levels: int) -> Path:
    """Write ``levels`` levels of full-sib mating; descendant path counts double per level."""
    rows = [{"id": 1, "sex": "F", "mother": -1, "father": -1}, {"id": 2, "sex": "M", "mother": -1, "father": -1}]
    for level in range(1, levels + 1):
        mother, father = 2 * level - 1, 2 * level
        rows.append({"id": 2 * level + 1, "sex": "F", "mother": mother, "father": father})
        rows.append({"id": 2 * level + 2, "sex": "M", "mother": mother, "father": father})
    return write_ped(path, rows)


def test_descendant_overflow_exits_1_before_expensive_phases(tmp_path):
    """Overflowing path counts end the run with one error line, before pair counting and F."""
    ped = _sib_mating_chain(tmp_path / "deep.tsv", levels=70)
    r = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "out")])

    assert r.returncode == 1
    assert "descendant path counts overflow int64 at max depth 70" in r.stderr
    assert "Traceback" not in r.stderr
    assert "relationship_counts" not in r.stderr
    assert "inbreeding (F" not in r.stderr
    assert not (tmp_path / "out" / "summary.yaml").exists()


def test_shallow_chain_still_summarizes(tmp_path):
    """The same structure short of the overflow runs to completion."""
    ped = _sib_mating_chain(tmp_path / "chain.tsv", levels=20)
    r = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "out")])

    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out" / "summary.yaml").exists()


def test_f_warning_depends_on_walk_budget(tmp_path, monkeypatch):
    """The F warning names rows and depth once the walk bound passes the budget, and only then."""
    quiet = run_pedsum(["summarize", "--in", str(EXAMPLE), "--out", str(tmp_path / "quiet")])
    assert quiet.returncode == 0
    assert "computing F on" not in quiet.stderr

    monkeypatch.setattr(pedsum.cli, "_F_WALK_WARN_VISITS", 0)
    loud = run_pedsum(["summarize", "--in", str(EXAMPLE), "--out", str(tmp_path / "loud")])
    assert loud.returncode == 0
    assert "WARNING" in loud.stderr
    assert "at max depth" in loud.stderr
    assert "--no-inbreeding" in loud.stderr

    skipped = run_pedsum(["summarize", "--in", str(EXAMPLE), "--out", str(tmp_path / "skip"), "--no-inbreeding"])
    assert "computing F on" not in skipped.stderr
