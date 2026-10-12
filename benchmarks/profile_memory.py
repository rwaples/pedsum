#!/usr/bin/env python3
"""Wall-time and peak-RSS profiler for any ``pedsum`` subcommand, with a regression gate.

Each repeat runs the command in its own child process. The child times its
``main()`` call (``cmd_s``), which leaves out interpreter startup and the import
of ``pedsum.cli``; the parent times the whole child (``wall_s``) and reads its
peak RSS from ``wait4``'s ``ru_maxrss``, the exact meter (in-process
``ru_maxrss`` only grows, so one process cannot measure several runs). Inside
the child a background thread samples ``/proc/self/statm`` every
~20 ms and attributes each sample to the ``timed`` block active when it was
taken, via ``pedsum.progress.current_profile_phase()``; those per-phase peaks are
approximate and diagnostic only.

The report gives the median and range of ``cmd_s``, ``wall_s`` and peak RSS
over the measured repeats. ``--compare BASELINE.json`` exits 1 when the exit codes
differ from the baseline's, or when the median ``cmd_s`` or the median peak
RSS exceeds 1.05x the baseline's; startup is a fixed cost that would dilute a
slowdown in the command itself. Compare medians of at
least 5 repeats; a single run is not evidence.

Usage::

    python benchmarks/profile_memory.py --label validate/1m --repeats 5 \
        --json-out /tmp/validate_before.json \
        -- validate --in /tmp/ped_1m_narrow.tsv --out /tmp/prof_out

    python benchmarks/profile_memory.py --label validate/1m --repeats 5 \
        --compare /tmp/validate_before.json \
        -- validate --in /tmp/ped_1m_narrow.tsv --out /tmp/prof_out

Everything after ``--`` is the verbatim pedsum argv. ``--out`` is rewritten to
a fresh directory per repeat. A ``<--in>.meta.json`` sidecar (from
``generate_pedigree.py``) is read when present to record rows / max depth /
extra-column count. Non-zero exit codes are recorded, not fatal, because
``validate`` exits 1 whenever it reports a finding.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_MiB = 1024 * 1024
_SAMPLE_INTERVAL_S = 0.02
_UNPROFILED = "(unprofiled)"
#: ``--compare`` fails above this ratio of the baseline's median.
GATE_RATIO = 1.05


def _read_rss_bytes() -> int:
    """Resident set size of this process in bytes, from ``/proc/self/statm``."""
    with open("/proc/self/statm") as fh:
        resident_pages = int(fh.read().split()[1])
    return resident_pages * _PAGE_SIZE


class _Sampler:
    """Background RSS sampler with incremental per-phase aggregation.

    Aggregates inline (peak / first-/last-seen per phase) instead of storing
    every sample, so the sampler's own footprint does not perturb the
    measurement it is taking.
    """

    def __init__(self, phase, interval: float = _SAMPLE_INTERVAL_S) -> None:
        self._phase = phase
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="rss-sampler", daemon=True)
        # phase label -> {"peak": bytes, "first_t": s, "last_t": s, "n": int}
        self.phases: dict[str, dict[str, float]] = {}

    def _record(self, t: float, phase: str | None, rss: int) -> None:
        key = phase if phase is not None else _UNPROFILED
        slot = self.phases.get(key)
        if slot is None:
            self.phases[key] = {"peak": rss, "first_t": t, "last_t": t, "n": 1}
        else:
            if rss > slot["peak"]:
                slot["peak"] = rss
            slot["last_t"] = t
            slot["n"] += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._record(time.perf_counter(), self._phase(), _read_rss_bytes())
            self._stop.wait(self._interval)

    def __enter__(self) -> _Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def _child(timing_out: Path, argv: list[str]) -> int:
    """Run ``pedsum <argv>`` in this process under the phase sampler; write its time and phases to ``timing_out``."""
    from pedsum.cli import main
    from pedsum.progress import current_profile_phase

    with _Sampler(current_profile_phase) as sampler:
        t0 = time.perf_counter()
        rc = main(argv)
        cmd_s = time.perf_counter() - t0
    timing_out.write_text(
        json.dumps(
            {
                "cmd_s": cmd_s,
                "per_phase": {
                    name: {
                        "peak_mib": slot["peak"] / _MiB,
                        "elapsed_s": slot["last_t"] - slot["first_t"],
                        "n_samples": int(slot["n"]),
                    }
                    for name, slot in sampler.phases.items()
                },
            }
        )
    )
    return rc


def _run_once(argv: list[str], out_dir: Path) -> dict:
    """Run one repeat in a child process; return its exit code, times, peak RSS and phases."""
    argv = [*argv]
    if "--out" in argv:
        argv[argv.index("--out") + 1] = str(out_dir)
    else:
        argv += ["--out", str(out_dir)]
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    timing_path = out_dir.parent / f".{out_dir.name}.timing.json"
    timing_path.unlink(missing_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, __file__, "--child", str(timing_path), "--", *argv])
    _, status, usage = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - t0
    exit_code = os.waitstatus_to_exitcode(status)
    if not timing_path.exists():
        raise SystemExit(f"pedsum {' '.join(argv)} did not return (exit {exit_code}); nothing to measure")
    timing = json.loads(timing_path.read_text())
    return {
        "exit_code": exit_code,
        "cmd_s": timing["cmd_s"],
        "wall_s": wall,
        # Linux reports ru_maxrss in KiB.
        "maxrss_mib": usage.ru_maxrss / 1024,
        "per_phase": timing["per_phase"],
    }


def _detect_metadata(argv: list[str]) -> dict:
    """The subcommand and the input's ``.meta.json`` in one dict."""
    in_path = Path(argv[argv.index("--in") + 1]) if "--in" in argv else None
    meta: dict = {}
    if in_path is not None:
        sidecar = in_path.with_suffix(in_path.suffix + ".meta.json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
    return {
        "subcommand": argv[0],
        "argv": argv,
        "input": str(in_path) if in_path else None,
        "rows": meta.get("rows"),
        "max_depth": meta.get("max_depth"),
        "extra_cols": meta.get("extra_cols"),
    }


def _aggregate(runs: list[dict]) -> dict:
    """Median + range of command time, wall time and peak RSS over measured runs; per-phase medians."""
    cmds = [r["cmd_s"] for r in runs]
    walls = [r["wall_s"] for r in runs]
    rss = [r["maxrss_mib"] for r in runs]
    phase_names = {name for r in runs for name in r["per_phase"]}
    per_phase: dict[str, dict] = {}
    for name in phase_names:
        present = [r["per_phase"][name] for r in runs if name in r["per_phase"]]
        per_phase[name] = {
            "peak_mib": statistics.median(p["peak_mib"] for p in present),
            "elapsed_s": statistics.median(p["elapsed_s"] for p in present),
            "n_runs": len(present),
        }
    return {
        "n_runs": len(runs),
        "exit_codes": sorted({r["exit_code"] for r in runs}),
        "cmd_s_median": statistics.median(cmds),
        "cmd_s_min": min(cmds),
        "cmd_s_max": max(cmds),
        "wall_s_median": statistics.median(walls),
        "wall_s_min": min(walls),
        "wall_s_max": max(walls),
        "maxrss_mib_median": statistics.median(rss),
        "maxrss_mib_min": min(rss),
        "maxrss_mib_max": max(rss),
        "per_phase": per_phase,
    }


def compare(agg: dict, baseline: dict) -> list[str]:
    """Regressions of ``agg`` against ``baseline``: other exit codes, or a median above ``GATE_RATIO`` x the baseline's.

    A run that exits differently did different work, so its times say nothing
    about the change.
    """
    failures = []
    if agg["exit_codes"] != baseline["exit_codes"]:
        line = f"exit_codes: {agg['exit_codes']} vs baseline {baseline['exit_codes']}"
        print(line)
        failures.append(line)
    for key, unit in (("cmd_s_median", "s"), ("maxrss_mib_median", " MiB")):
        ratio = agg[key] / baseline[key]
        line = f"{key}: {agg[key]:.2f}{unit} vs baseline {baseline[key]:.2f}{unit} ({ratio:.3f}x)"
        print(line)
        if ratio > GATE_RATIO:
            failures.append(line)
    return failures


def _print_report(label: str, meta: dict, agg: dict) -> None:
    print(f"\n=== {label} ===")
    print(
        f"{meta['subcommand']} input={meta['input']} rows={meta['rows']} max_depth={meta['max_depth']} "
        f"extra_cols={meta['extra_cols']} exit_codes={agg['exit_codes']}"
    )
    print(
        f"command: median {agg['cmd_s_median']:.2f}s (range {agg['cmd_s_min']:.2f}-{agg['cmd_s_max']:.2f}); "
        f"process wall: median {agg['wall_s_median']:.2f}s; "
        f"peak RSS (wait4): median {agg['maxrss_mib_median']:.1f} MiB "
        f"(range {agg['maxrss_mib_min']:.1f}-{agg['maxrss_mib_max']:.1f}) over {agg['n_runs']} run(s)"
    )
    print("per-phase sampled peak RSS (sorted):")
    for name, p in sorted(agg["per_phase"].items(), key=lambda kv: kv[1]["peak_mib"], reverse=True):
        print(f"  {p['peak_mib']:8.1f} MiB  {p['elapsed_s']:6.2f}s  {name}")


def _parse_self_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Split this script's own options from the trailing pedsum argv."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", help="cell label, e.g. validate/1m")
    p.add_argument("--repeats", type=int, default=5, help="measured repeats (default: 5)")
    p.add_argument("--warmup", type=int, default=1, help="warm-up runs before measuring (default: 1)")
    p.add_argument(
        "--compare", type=Path, default=None, help="JSON from a prior run; exit 1 on a >5%% median regression"
    )
    p.add_argument("--json-out", type=Path, default=None, help="write the aggregated metrics as JSON here")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/pedsum_profile"),
        help="base directory for per-repeat output dirs (default: /tmp/pedsum_profile)",
    )
    p.add_argument("--child", type=Path, default=None, help=argparse.SUPPRESS)
    args, rest = p.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    return args, rest


def main(argv: list[str] | None = None) -> int:
    """Profile a pedsum command over warm-up + measured repeats; print a report and gate on ``--compare``."""
    args, rest = _parse_self_args(argv)
    if args.child is not None:
        return _child(args.child, rest)
    if not rest or args.label is None:
        raise SystemExit("expected: profile_memory.py --label L [opts] -- <subcommand> --in ... [flags]")

    meta = _detect_metadata(rest)
    work = args.work_dir / args.label.replace("/", "_")
    for w in range(args.warmup):
        _run_once(rest, work / f"warmup_{w}")
    runs = [_run_once(rest, work / f"run_{r}") for r in range(args.repeats)]

    agg = _aggregate(runs)
    _print_report(args.label, meta, agg)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"label": args.label, "metadata": meta, "aggregate": agg, "runs": runs}, indent=2)
        )
        print(f"\nwrote {args.json_out}")
    if args.compare is not None:
        failures = compare(agg, json.loads(args.compare.read_text())["aggregate"])
        for line in failures:
            print(f"REGRESSION: {line}")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
