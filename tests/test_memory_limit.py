"""The best-effort memory limit (ADR 0004 §3) and atomic output publication."""

from __future__ import annotations

import argparse
import functools
import logging
import subprocess
import sys
import threading
from pathlib import Path

import polars as pl
import pytest
import yaml
from conftest import EXAMPLE, run_pedsum

from pedsum import report
from pedsum.memory import EXIT_MEMORY_LIMIT, MemoryWatchdog, memory_headroom_bytes, parse_size, resolve_limit
from pedsum.report import _to_csv_gz, _write_long_tsv, _write_yaml, atomic_output

GiB = 2**30


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12G", 12 * GiB),
        ("500M", 500 * 2**20),
        ("500mb", 500 * 2**20),
        ("1.5G", int(1.5 * GiB)),
        ("0", 0),
        ("4096", 4096),
    ],
)
def test_parse_size(text, expected):
    """Binary suffixes, an optional B, and a plain byte count all parse; 0 means off."""
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["lots", "-1G", "G"])
def test_parse_size_rejects(text):
    """Text, a negative size, and a bare unit are usage errors."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_size(text)


def _fake_roots(
    tmp_path: Path, *, mem_available_kb: int | None, levels: dict[str, tuple[str, int]]
) -> tuple[Path, Path]:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    if mem_available_kb is not None:
        (proc / "meminfo").write_text(f"MemTotal: 99999999 kB\nMemAvailable: {mem_available_kb} kB\n")
    (proc / "self" / "cgroup").write_text("0::/user.slice/app.scope\n")
    cgroup = tmp_path / "cgroup"
    for rel, (limit, current) in levels.items():
        level = cgroup / rel
        level.mkdir(parents=True, exist_ok=True)
        (level / "memory.max").write_text(f"{limit}\n")
        (level / "memory.current").write_text(f"{current}\n")
    return proc, cgroup


def test_headroom_parent_cgroup_binds_when_leaf_is_max(tmp_path):
    """A parent scope's memory.max binds when the leaf reads max."""
    proc, cgroup = _fake_roots(
        tmp_path,
        mem_available_kb=8 * 2**20,
        levels={"user.slice": (str(4 * GiB), GiB), "user.slice/app.scope": ("max", GiB)},
    )
    assert memory_headroom_bytes(proc, cgroup) == 3 * GiB


def test_headroom_leaf_cgroup_binds(tmp_path):
    """The tightest level wins when the leaf has its own limit."""
    proc, cgroup = _fake_roots(
        tmp_path,
        mem_available_kb=8 * 2**20,
        levels={"user.slice": (str(4 * GiB), GiB), "user.slice/app.scope": (str(2 * GiB), GiB)},
    )
    assert memory_headroom_bytes(proc, cgroup) == GiB


def test_headroom_host_binds_without_cgroup_limits(tmp_path):
    """Host MemAvailable is the headroom when no cgroup level sets a limit."""
    proc, cgroup = _fake_roots(tmp_path, mem_available_kb=2**20, levels={"user.slice/app.scope": ("max", GiB)})
    assert memory_headroom_bytes(proc, cgroup) == GiB


def test_headroom_none_when_nothing_readable(tmp_path):
    """No /proc and no cgroup files means no default limit."""
    assert memory_headroom_bytes(tmp_path / "no-proc", tmp_path / "no-cgroup") is None


def test_resolve_limit():
    """0 turns the limit off, an explicit size is used as given."""
    assert resolve_limit(0) is None
    assert resolve_limit(5 * GiB) == 5 * GiB
    default = resolve_limit(None)
    assert default is None or default > 0


def test_disabled_limit_starts_no_thread():
    """An inactive watchdog starts no thread and disarm() still works."""
    with MemoryWatchdog(None) as wd:
        assert wd._thread is None
        assert not any(t.name == "pedsum-memory-watchdog" for t in threading.enumerate())
        with wd.disarm():
            pass


def test_breach_runs_callback_logs_and_exits_3(caplog):
    """A breach runs the callback, logs the phase at ERROR, and exits 3."""
    exits: list[int] = []
    breaches: list[tuple[int, int]] = []
    wd = MemoryWatchdog(GiB, poll_s=0.01, rss_reader=lambda: 2 * GiB, exit_fn=exits.append, phase=lambda: "kinship DP")
    wd.on_breach(lambda rss, limit: breaches.append((rss, limit)))
    with caplog.at_level(logging.INFO), wd:
        assert wd._thread is not None
        wd._thread.join(timeout=5)
    assert exits == [EXIT_MEMORY_LIMIT]
    assert breaches == [(2 * GiB, GiB)]
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("kinship DP used 2.0 GiB RSS, over the 1.0 GiB limit; stopping" in m for m in errors)


def test_failing_callback_is_logged_and_skipped(caplog):
    """A raising callback is logged; later callbacks and the exit still run."""
    exits: list[int] = []
    ran: list[str] = []

    def boom(rss, limit):
        raise RuntimeError("disk full")

    wd = MemoryWatchdog(GiB, poll_s=0.01, rss_reader=lambda: 2 * GiB, exit_fn=exits.append)
    wd.on_breach(boom)
    wd.on_breach(lambda rss, limit: ran.append("second"))
    with caplog.at_level(logging.INFO), wd:
        assert wd._thread is not None
        wd._thread.join(timeout=5)
    assert ran == ["second"]
    assert exits == [EXIT_MEMORY_LIMIT]
    assert any("callback" in r.getMessage() and r.exc_info for r in caplog.records)


def test_breach_after_disarm_warns_and_does_not_exit(caplog):
    """A breach waiting on the publication lock only warns once the main thread published."""
    exits: list[int] = []
    rss = [0]
    sampled_high = threading.Event()

    def reader() -> int:
        if rss[0]:
            sampled_high.set()
        return rss[0]

    wd = MemoryWatchdog(GiB, poll_s=0.01, rss_reader=reader, exit_fn=exits.append)
    with caplog.at_level(logging.INFO), wd:
        with wd.disarm():
            rss[0] = 2 * GiB
            assert sampled_high.wait(timeout=5)
            # The breach path is now blocked on the publication lock.
            assert wd._thread is not None
            assert wd._thread.is_alive()
        wd._thread.join(timeout=5)
    assert exits == []
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("after its output was published" in m for m in warnings)


def test_real_breach_in_subprocess_exits_3():
    """A real allocation past the limit stops the process with exit 3 and an ERROR line."""
    code = (
        "import logging, time\n"
        "import numpy as np\n"
        "from pedsum.memory import MemoryWatchdog\n"
        "logging.basicConfig(level=logging.INFO)\n"
        "with MemoryWatchdog(200 * 2**20, poll_s=0.05):\n"
        "    block = np.ones(400 * 2**20, dtype=np.uint8)\n"
        "    time.sleep(10)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == EXIT_MEMORY_LIMIT, res.stderr
    assert "ERROR" in res.stderr
    assert "over the 0.2 GiB limit; stopping" in res.stderr


def _no_partials(directory: Path) -> bool:
    return not any(p.name.startswith(".") and ".partial-" in p.name for p in directory.iterdir())


def test_atomic_output_replaces_on_success(tmp_path):
    """The target keeps its old content until the block exits, then holds the new."""
    target = tmp_path / "out.txt"
    target.write_text("old")
    with atomic_output(target) as tmp:
        tmp.write_text("new")
        assert target.read_text() == "old"
    assert target.read_text() == "new"
    assert _no_partials(tmp_path)


def test_atomic_output_keeps_previous_on_error(tmp_path):
    """An exception in the block leaves the old content and no temporary."""
    target = tmp_path / "out.txt"
    target.write_text("old")

    def write_half() -> None:
        with atomic_output(target) as tmp:
            tmp.write_text("half")
            raise RuntimeError("stopped mid-write")

    with pytest.raises(RuntimeError):
        write_half()
    assert target.read_text() == "old"
    assert _no_partials(tmp_path)


def _raise_after_partial_csv(self, file, *args, **kwargs):
    data = b"id\tsex\n1\t0\n"
    if isinstance(file, (str, Path)):
        Path(file).write_bytes(data)
    else:
        file.write(data)
    raise RuntimeError("serialiser failed part-way")


def _raise_after_partial_yaml(data, stream, **kwargs):
    stream.write("partial: [")
    raise RuntimeError("serialiser failed part-way")


@pytest.mark.parametrize("writer", ["yaml", "long_tsv", "csv_gz_pigz", "csv_gz_gzip"])
def test_writer_failure_keeps_previous_output(tmp_path, monkeypatch, writer):
    """A serialiser failing part-way leaves the previous output byte for byte."""
    target = tmp_path / {"yaml": "s.yaml", "long_tsv": "s.tsv"}.get(writer, "s.tsv.gz")
    previous = b"previous run's bytes\n"
    target.write_bytes(previous)
    df = pl.DataFrame({"id": [1, 2], "sex": [0, 1]})

    if writer == "yaml":
        monkeypatch.setattr(report.yaml, "safe_dump", _raise_after_partial_yaml)
        call = functools.partial(_write_yaml, {"a": 1}, target)
    else:
        monkeypatch.setattr(pl.DataFrame, "write_csv", _raise_after_partial_csv)
        if writer == "long_tsv":
            call = functools.partial(_write_long_tsv, {"a": 1}, target)
        else:
            if writer == "csv_gz_gzip":
                monkeypatch.setattr(report.shutil, "which", lambda _name: None)
            elif report.shutil.which("pigz") is None:
                pytest.skip("pigz not on PATH")
            call = functools.partial(_to_csv_gz, df, target)

    with pytest.raises(RuntimeError, match="part-way"):
        call()
    assert target.read_bytes() == previous
    assert _no_partials(tmp_path)


def test_writers_publish_parseable_output(tmp_path):
    """A successful YAML write is complete and leaves no temporary."""
    _write_yaml({"a": 1.23456}, tmp_path / "s.yaml")
    assert yaml.safe_load((tmp_path / "s.yaml").read_text()) == {"a": 1.2346}
    assert _no_partials(tmp_path)


def test_cli_accepts_max_memory_zero(tmp_path):
    """--max-memory 0 logs that the limit is off and starts no thread."""
    res = run_pedsum(["validate", "--in", str(EXAMPLE), "--out", str(tmp_path / "out"), "--max-memory", "0"])
    assert res.returncode in (0, 1), res.stderr
    assert "memory limit off" in res.stderr
    assert not any(t.name == "pedsum-memory-watchdog" for t in threading.enumerate())


def test_cli_logs_default_limit(tmp_path):
    """A run without --max-memory logs the limit it resolved."""
    res = run_pedsum(["validate", "--in", str(EXAMPLE), "--out", str(tmp_path / "out")])
    assert res.returncode in (0, 1), res.stderr
    assert "memory limit" in res.stderr
