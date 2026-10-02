"""The tqdm bar :func:`pedsum.progress.relationship_progress` drives."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pedigree_graph._progress
import pytest
from pedigree_graph import RelationshipProgress

import pedigree_summary as ps
from pedsum.progress import relationship_progress

EXAMPLE = Path(__file__).resolve().parents[1] / "example_pedigree.tsv"

_REPORTS = (
    RelationshipProgress("preparing", 0, None, 1.0),
    RelationshipProgress("walking", 0, 2_000, 2.0),
    RelationshipProgress("walking", 280, 2_000, 30.0),
    RelationshipProgress("walking", 1_990, 2_000, 1_700.0),
    RelationshipProgress("finishing", 2_000, 2_000, 1_833.0),
)


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def info_logging():
    """The ``pedigree_graph`` logger at INFO, as in a run without ``--quiet``."""
    # setLevel, not monkeypatch: it clears the loggers' isEnabledFor cache.
    pg_logger = logging.getLogger("pedigree_graph")
    old = pg_logger.level
    pg_logger.setLevel(logging.INFO)
    yield pg_logger
    pg_logger.setLevel(old)


def _frames(stream: io.StringIO) -> list[str]:
    return [f.rstrip("\n") for f in stream.getvalue().split("\r") if f.strip()]


def test_bar_follows_each_phase_to_the_total(info_logging):
    """Each report redraws the bar in its phase's form, ending at the total."""
    stream = _Terminal()
    frames = []
    with relationship_progress("relationship_counts", stream) as progress:
        assert progress is not None
        for report in _REPORTS:
            progress(report)
            frames.append(_frames(stream)[-1])

    assert frames[0].startswith("relationship_counts: preparing [")
    assert frames[2].startswith("relationship_counts  14%|")
    assert "280/2,000 rows" in frames[2]
    assert frames[4].startswith("relationship_counts: assembling 100%|")
    assert "2,000/2,000 rows" in frames[4]
    assert stream.getvalue().endswith("\n"), "bar closed"


def test_bar_has_no_rate_or_remaining_time(info_logging):
    """Row cost is too uneven for a rate or ETA, so the bar shows neither."""
    stream = _Terminal()
    with relationship_progress("relationship_counts", stream) as progress:
        for report in _REPORTS:
            progress(report)
    text = stream.getvalue()
    assert "<" not in text
    assert "/s" not in text
    assert "?" not in text


def test_completed_call_fills_a_bar_left_short(info_logging):
    """A call that returns between two reports still ends on a full bar."""
    stream = _Terminal()
    with relationship_progress("relationship_counts", stream) as progress:
        progress(_REPORTS[3])
    assert _frames(stream)[-1].startswith("relationship_counts 100%|")
    assert "2,000/2,000 rows" in _frames(stream)[-1]


def test_bar_closes_where_it_stopped_when_the_call_raises(info_logging):
    """A cancelled call closes the bar at the last report, not at the total."""
    stream = _Terminal()

    def cancelled_call():
        with relationship_progress("relationship_counts", stream) as progress:
            progress(_REPORTS[2])
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cancelled_call()
    assert "280/2,000 rows" in _frames(stream)[-1]
    assert stream.getvalue().endswith("\n"), "bar closed"


def test_short_call_draws_nothing(info_logging):
    """No report arrives before a sub-second call returns, so no bar is drawn."""
    stream = _Terminal()
    with relationship_progress("relationship_counts", stream) as progress:
        assert progress is not None
    assert stream.getvalue() == ""


def test_log_lines_go_above_the_bar(info_logging, monkeypatch):
    """A log line on stderr clears the bar, prints, and the bar is redrawn."""
    stream = _Terminal()
    monkeypatch.setattr(sys, "stderr", stream)
    handler = logging.StreamHandler(sys.stderr)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with relationship_progress("relationship_counts") as progress:
            progress(_REPORTS[2])
            logging.getLogger("pedigree_graph").info("total: 7 pairs")
    finally:
        root.removeHandler(handler)
    before, after = stream.getvalue().split("total: 7 pairs\n")
    assert before.rsplit("\r", 2)[-2].strip() == "", "bar line cleared before the log line"
    assert after.startswith("\rrelationship_counts  14%|"), "bar redrawn below the log line"


def test_not_a_terminal_keeps_default_logging(info_logging):
    """A redirected stderr gets pedigree-graph's 30 s log lines instead."""
    with relationship_progress("relationship_counts", io.StringIO()) as progress:
        assert progress is None


def test_quiet_keeps_default_logging(info_logging):
    """``--quiet`` raises the level past INFO, which hides the bar too."""
    info_logging.setLevel(logging.WARNING)
    with relationship_progress("relationship_counts", _Terminal()) as progress:
        assert progress is None


def test_counts_unchanged_under_the_bar(info_logging, monkeypatch):
    """Real reports through the bar leave the counts as they are without it."""
    df, _ = ps.load_and_validate(EXAMPLE)
    pg = ps._build_pedigree_graph(df)
    expected = pg.relationship_counts(max_degree=5)
    # A tick far below the call's length, so the bar sees real reports.
    monkeypatch.setattr(pedigree_graph._progress, "TICK_S", 1e-4)
    with relationship_progress("relationship_counts", _Terminal()) as progress:
        got = pg.relationship_counts(max_degree=5, progress=progress)
    assert got == expected
