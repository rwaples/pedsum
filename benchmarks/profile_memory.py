#!/usr/bin/env python3
"""In-process peak-RSS profiler for ``pedsum summarize``.

The memory-reduction plan is *measure-first*: no optimization is merged before
its before/after RSS delta is shown. This profiler is that gate. It runs
``pedsum.cli._run_summarize`` in this process (reusing ``_parse_args`` so flag
parsing matches the real CLI) while a background thread samples this process's
RSS from ``/proc/self/statm`` every ~20 ms, then reports:

* overall peak RSS (median across measured repeats, with observed range), and
* per-phase peak RSS, attributed via ``pedsum.cli._current_profile_phase()`` —
  the label of the ``_timed`` block active when each sample was taken.

Secondary metadata: total runtime and approximate per-phase elapsed time. A
runtime regression (total or the targeted phase worsening >20% vs ``--baseline``)
is *flagged*, not blocked — the operator decides whether the RSS win is worth it.

This intentionally prioritizes phase attribution over full CLI startup/import
RSS; the interpreter and imports are already resident before sampling starts.

Usage::

    python benchmarks/profile_memory.py --label narrow/streaming --repeats 3 \
        -- summarize --in /tmp/ped_narrow.tsv --out /tmp/prof_out \
        --birth-year-col birth_year

Everything after ``--`` is the verbatim ``summarize`` argv. ``--out`` is
rewritten to a fresh directory per repeat. A ``<--in>.meta.json`` sidecar (from
``generate_pedigree.py``) is read when present to record rows / max depth /
extra-column count.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

# Make ``pedsum`` importable when this script is run as ``python
# benchmarks/profile_memory.py`` from the repo root (benchmarks/ is sys.path[0],
# not the repo root). Mirrors the conftest.py bootstrap the tests rely on.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsum.cli import _current_profile_phase, _parse_args, _run_summarize

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_MiB = 1024 * 1024
_SAMPLE_INTERVAL_S = 0.02
_UNPROFILED = "(unprofiled)"


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

    def __init__(self, interval: float = _SAMPLE_INTERVAL_S) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="rss-sampler", daemon=True)
        self.global_peak = 0
        # phase label -> {"peak": bytes, "first_t": s, "last_t": s, "n": int}
        self.phases: dict[str, dict[str, float]] = {}

    def _record(self, t: float, phase: str | None, rss: int) -> None:
        if rss > self.global_peak:
            self.global_peak = rss
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
            self._record(time.perf_counter(), _current_profile_phase(), _read_rss_bytes())
            self._stop.wait(self._interval)

    def __enter__(self) -> _Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def _run_once(summarize_argv: list[str], out_dir: Path) -> dict:
    """Profile one ``summarize`` run into ``out_dir``; return its metrics dict."""
    argv = [*summarize_argv]
    # Rewrite --out to this run's fresh directory.
    if "--out" in argv:
        argv[argv.index("--out") + 1] = str(out_dir)
    else:
        argv += ["--out", str(out_dir)]
    args = _parse_args(argv)
    cmd = "python pedigree_summary.py " + " ".join(argv)

    gc.collect()
    t0 = time.perf_counter()
    with _Sampler() as sampler:
        rc = _run_summarize(args, cmd)
    total_runtime = time.perf_counter() - t0
    if rc != 0:
        raise SystemExit(f"summarize returned non-zero exit code {rc}; argv={argv}")

    per_phase = {
        name: {
            "peak_mib": slot["peak"] / _MiB,
            "elapsed_s": slot["last_t"] - slot["first_t"],
            "n_samples": int(slot["n"]),
        }
        for name, slot in sampler.phases.items()
    }
    return {
        "global_peak_mib": sampler.global_peak / _MiB,
        "total_runtime_s": total_runtime,
        "per_phase": per_phase,
    }


def _detect_metadata(summarize_argv: list[str]) -> dict:
    """Pull engine / flag state and the input's .meta.json into one dict."""
    in_path = None
    if "--in" in summarize_argv:
        in_path = Path(summarize_argv[summarize_argv.index("--in") + 1])
    meta: dict = {}
    if in_path is not None:
        sidecar = in_path.with_suffix(in_path.suffix + ".meta.json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
    return {
        "input": str(in_path) if in_path else None,
        "rows": meta.get("rows"),
        "max_depth": meta.get("max_depth"),
        "extra_cols": meta.get("extra_cols"),
        "engine": "matrix" if "--per-individual-pairs" in summarize_argv else "streaming",
        "inbreeding": "--no-inbreeding" not in summarize_argv,
        "effective_size": "--no-effective-size" not in summarize_argv,
    }


def _aggregate(runs: list[dict]) -> dict:
    """Median + range over measured runs; per-phase peak/elapsed medians."""
    peaks = [r["global_peak_mib"] for r in runs]
    runtimes = [r["total_runtime_s"] for r in runs]
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
        "peak_mib_median": statistics.median(peaks),
        "peak_mib_min": min(peaks),
        "peak_mib_max": max(peaks),
        "runtime_s_median": statistics.median(runtimes),
        "per_phase": per_phase,
    }


def _flag_regressions(label: str, agg: dict, baseline: dict | None) -> list[str]:
    """Return >20% runtime-regression warnings vs a prior baseline (if any)."""
    if baseline is None:
        return []
    flags: list[str] = []
    base_rt = baseline.get("runtime_s_median")
    if base_rt and agg["runtime_s_median"] > 1.2 * base_rt:
        flags.append(f"[{label}] total runtime {agg['runtime_s_median']:.2f}s > 1.2x baseline {base_rt:.2f}s")
    base_phases = baseline.get("per_phase", {})
    for name, cur in agg["per_phase"].items():
        bp = base_phases.get(name)
        if bp and bp.get("elapsed_s", 0) > 0 and cur["elapsed_s"] > 1.2 * bp["elapsed_s"]:
            flags.append(f"[{label}] phase {name!r} {cur['elapsed_s']:.2f}s > 1.2x baseline {bp['elapsed_s']:.2f}s")
    return flags


def _print_report(label: str, meta: dict, agg: dict, flags: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(
        f"input={meta['input']} rows={meta['rows']} max_depth={meta['max_depth']} "
        f"extra_cols={meta['extra_cols']} engine={meta['engine']} "
        f"inbreeding={meta['inbreeding']} effective_size={meta['effective_size']}"
    )
    print(
        f"peak RSS: median {agg['peak_mib_median']:.1f} MiB "
        f"(range {agg['peak_mib_min']:.1f}-{agg['peak_mib_max']:.1f}); "
        f"total runtime median {agg['runtime_s_median']:.2f}s"
    )
    print("per-phase peak RSS (sorted):")
    for name, p in sorted(agg["per_phase"].items(), key=lambda kv: kv[1]["peak_mib"], reverse=True):
        print(f"  {p['peak_mib']:8.1f} MiB  {p['elapsed_s']:6.2f}s  {name}")
    for f in flags:
        print(f"REGRESSION FLAG: {f}")


def _parse_self_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Split this script's own options from the trailing ``summarize`` argv."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="cell label, e.g. narrow/streaming")
    p.add_argument("--repeats", type=int, default=3, help="measured repeats (default: 3)")
    p.add_argument("--warmup", type=int, default=1, help="warm-up runs before measuring (default: 1)")
    p.add_argument("--baseline", type=Path, default=None, help="JSON from a prior run to flag regressions against")
    p.add_argument("--json-out", type=Path, default=None, help="write the aggregated metrics as JSON here")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/pedsum_profile"),
        help="base directory for per-repeat output dirs (default: /tmp/pedsum_profile)",
    )
    self_argv = argv if argv is not None else None
    return p.parse_known_args(self_argv)


def main(argv: list[str] | None = None) -> int:
    """Profile a summarize cell over warm-up + measured repeats; print a report."""
    args, rest = _parse_self_args(argv)
    # argparse leaves the trailing "summarize ..." (after --) in `rest`.
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest or rest[0] != "summarize":
        raise SystemExit("expected: profile_memory.py [opts] -- summarize --in ... [flags]")

    meta = _detect_metadata(rest)
    work = args.work_dir / args.label.replace("/", "_")

    for w in range(args.warmup):
        _run_once(rest, work / f"warmup_{w}")
        gc.collect()

    runs: list[dict] = []
    for r in range(args.repeats):
        runs.append(_run_once(rest, work / f"run_{r}"))
        gc.collect()

    agg = _aggregate(runs)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    base_agg = baseline.get("aggregate") if baseline else None
    flags = _flag_regressions(args.label, agg, base_agg)
    _print_report(args.label, meta, agg, flags)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {"label": args.label, "metadata": meta, "aggregate": agg, "runs": runs, "flags": flags},
                indent=2,
            )
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
