#!/usr/bin/env python3
r"""Paired wall time and peak memory of ``pedsum assortative-mating`` in two trees.

``run`` measures tree A (``--a-python``, ``--a-script``) against tree B at each
``--point``. Each pair runs both trees once on the same input, alternating
which goes first (ABBA...), so drift on a shared machine falls on both sides.
Every run is a fresh process in its own transient systemd scope:
``/usr/bin/time`` gives wall seconds and ``ru_maxrss``, and the scope's cgroup
``memory.peak`` gives the peak of the whole process tree. After each pair, the
``draws_used`` of every permutation test and the valid and failed bootstrap
draws must agree, so both trees did the same work.

Before the pairs, each tree runs once with an empty ``NUMBA_CACHE_DIR``: the
first run after an install, recorded as ``cold`` and not paired. The pairs then
run with each tree's default numba cache, which that run primed.

Configurations:

* ``a``: one binary trait (``--trait dx``), unstratified.
* ``b``: ``--trait liab dx --stratify-by birth_year`` (all four cell types).

Inputs come from ``generate_assortative_mating.py``. Usage::

    python benchmarks/bench_assortative_mating.py run \
        --a-python OLD/.pixi/envs/default/bin/python --a-script OLD/pedigree_summary.py \
        --b-python .pixi/envs/default/bin/python --b-script pedigree_summary.py \
        --pairs 10 --threads 6 --out /tmp/am_bench.jsonl --point b:/tmp/am_1e5.tsv:999:0
    python benchmarks/bench_assortative_mating.py summary /tmp/am_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

CONFIG_ARGS = {
    "a": ["--trait", "dx"],
    "b": ["--trait", "liab", "dx", "--stratify-by", "birth_year", "--birth-year-col", "birth_year"],
}
# Runs inside the scope: time the CLI, then report the scope's cgroup peak.
SCOPE = (
    'cg=$(cut -d: -f3 /proc/self/cgroup); /usr/bin/time -f "%e %M" "$@"; status=$?; '
    'echo "memory_peak $(cat /sys/fs/cgroup$cg/memory.peak)" >&2; exit $status'
)
GATE = 1.05
# The CLI's per-stage INFO lines (``cli._timed``).
STAGE = re.compile(r"(load\+validate|assortative mating) in ([0-9.]+)s")


def draws(payload: dict) -> dict[str, list[int]]:
    """``draws_used`` of every permutation test and the bootstrap's valid and failed draws, by estimate."""
    out = {}
    for cell in payload["mate_correlation"]:
        for form in ("crude", "stratified"):
            for name, rec in cell.get(form, {}).items():
                if not isinstance(rec, dict):
                    continue
                key = f"{cell['mother']}x{cell['father']}.{form}.{name}"
                if "permutations" in rec:
                    out[f"{key}.permutations"] = [rec["permutations"]["draws_used"]]
                if "bootstrap" in rec:
                    out[f"{key}.bootstrap"] = [rec["bootstrap"]["valid"], rec["bootstrap"]["failed"]]
    return out


def run_once(python: Path, script: Path, point: dict, threads: int, cache_dir: str | None = None) -> dict:
    """One CLI run in its own scope: wall seconds, ``ru_maxrss`` and ``memory.peak`` in bytes, and its draws."""
    env = dict(os.environ)
    if cache_dir is not None:
        env["NUMBA_CACHE_DIR"] = cache_dir
    with tempfile.TemporaryDirectory() as out_dir:
        cli = [
            str(python), str(script), "assortative-mating", "--in", str(point["data"]), "--out", out_dir,
            *CONFIG_ARGS[point["config"]], "--permutations", str(point["permutations"]),
            "--bootstrap", str(point["bootstrap"]), "--threads", str(threads),
        ]  # fmt: skip
        cmd = ["systemd-run", "--user", "--scope", "-q", "--", "bash", "-c", SCOPE, "bash", *cli]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=script.parent, env=env)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"run failed ({proc.returncode}): {' '.join(cli)}")
        payload = yaml.safe_load((Path(out_dir) / "assortative_mating.yaml").read_text())["assortative_mating"]
    timing, peak = proc.stderr.strip().splitlines()[-2:]
    wall, rss_kb = timing.split()
    return {
        "wall_s": float(wall),
        "max_rss": int(rss_kb) * 1024,
        "memory_peak": int(peak.split()[1]),
        "threads_used": payload["settings"]["threads"],
        "stages_s": {m[1]: float(m[2]) for m in STAGE.finditer(proc.stderr)},
        "draws": draws(payload),
    }


def run(args: argparse.Namespace) -> None:
    """Run every ``--point``: the cold runs, then the pairs; append the records to ``--out``."""
    trees = {"a": (args.a_python, args.a_script.resolve()), "b": (args.b_python, args.b_script.resolve())}
    for spec in args.point:
        config, data, perms, boots = spec.rsplit(":", 3)
        data_path = Path(data)
        point = {
            "config": config,
            "data": data_path,
            "pairs": json.loads(data_path.with_suffix(data_path.suffix + ".meta.json").read_text())["pairs"],
            "permutations": int(perms),
            "bootstrap": int(boots),
        }
        base = {k: v for k, v in point.items() if k != "data"} | {"threads": args.threads}
        records = []
        if args.cold:
            for side, (python, script) in trees.items():
                with tempfile.TemporaryDirectory() as cache:
                    r = run_once(python, script, point, args.threads, cache_dir=cache)
                records.append(base | {"kind": "cold", "side": side, "load1": os.getloadavg()[0]} | r)
                run_once(python, script, point, args.threads)
        for i in range(args.pairs):
            order = ("a", "b") if i % 2 == 0 else ("b", "a")
            runs = {side: run_once(*trees[side], point, args.threads) for side in order}
            if runs["a"]["draws"] != runs["b"]["draws"]:
                raise SystemExit(f"draws differ at {spec} pair {i}: {runs['a']['draws']} vs {runs['b']['draws']}")
            records.append(
                base
                | {"kind": "pair", "pair": i, "first": order[0], "load1": os.getloadavg()[0]}
                | {f"{side}_{k}": v for side, r in runs.items() for k, v in r.items() if k != "draws"}
            )
            print(
                f"{config} pairs={point['pairs']} {perms}/{boots} t{args.threads} #{i}: "
                f"wall {runs['a']['wall_s']:.2f} / {runs['b']['wall_s']:.2f} s, "
                f"peak {runs['a']['memory_peak'] / 2**20:.0f} / {runs['b']['memory_peak'] / 2**20:.0f} MiB",
                flush=True,
            )
        with args.out.open("a") as fh:
            for r in records:
                fh.write(json.dumps(r | {"finished": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def summary(args: argparse.Namespace) -> int:
    """Print the paired medians and ratios (B / A) as Markdown and return 1 if a median ratio exceeds the gate."""
    records = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    key = lambda r: (r["config"], r["pairs"], r["permutations"], r["bootstrap"], r["threads"])  # noqa: E731
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r["kind"] == "pair":
            groups.setdefault(key(r), []).append(r)
    print(
        "| config | Mating Pairs | P/B | threads | pairs | wall s A / B | wall B/A [min, max] | peak MiB A / B | peak B/A [min, max] | max load |"
    )
    print("|---|---:|---|---:|---:|---|---|---|---|---:|")  # fmt: skip
    worst = {"wall_s": 0.0, "memory_peak": 0.0}
    for k in sorted(groups, key=lambda k: (k[1], k[0], k[2:], k[4])):
        rows = groups[k]
        cells = [k[0], f"{k[1]:,}", f"{k[2]}/{k[3]}", str(k[4]), str(len(rows))]
        for field, scale, fmt in (("wall_s", 1, "{:.2f}"), ("memory_peak", 2**20, "{:.0f}")):
            ratios = [r[f"b_{field}"] / r[f"a_{field}"] for r in rows]
            median = statistics.median(ratios)
            worst[field] = max(worst[field], median)
            sides = " / ".join(fmt.format(statistics.median(r[f"{s}_{field}"] for r in rows) / scale) for s in "ab")
            cells += [sides, f"{median:.3f} [{min(ratios):.3f}, {max(ratios):.3f}]"]
        cells.append(f"{max(r['load1'] for r in rows):.1f}")
        print("| " + " | ".join(cells) + " |")
    cold = [r for r in records if r["kind"] == "cold"]
    if cold:
        print("\nFirst run with an empty numba cache (not gated):\n")
        print("| config | Mating Pairs | threads | side | wall s | peak MiB |")
        print("|---|---:|---:|---|---:|---:|")
        for r in sorted(cold, key=lambda r: (*key(r), r["side"])):
            print(
                f"| {r['config']} | {r['pairs']:,} | {r['threads']} | {r['side']} | {r['wall_s']:.2f} "
                f"| {r['memory_peak'] / 2**20:.0f} |"
            )
    verdict = "PASS" if max(worst.values()) <= GATE else "FAIL"
    print(f"\nWorst median ratio: wall {worst['wall_s']:.3f}, peak {worst['memory_peak']:.3f}. Gate {GATE}: {verdict}.")
    return 0 if verdict == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``run`` or ``summary``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="measure tree A against tree B")
    for side in ("a", "b"):
        r.add_argument(f"--{side}-python", type=Path, required=True, help=f"tree {side.upper()}'s interpreter")
        r.add_argument(f"--{side}-script", type=Path, required=True, help=f"tree {side.upper()}'s pedigree_summary.py")
    r.add_argument("--point", action="append", required=True, metavar="CONFIG:DATA:PERMS:BOOTS")
    r.add_argument("--pairs", type=int, default=10, help="ABAB pairs per point (default 10)")
    r.add_argument("--threads", type=int, default=1, help="--threads of every run (default 1)")
    r.add_argument("--no-cold", dest="cold", action="store_false", help="skip the empty-cache first run")
    r.add_argument("--out", type=Path, required=True, help="JSONL, appended")
    s = sub.add_parser("summary", help="print a JSONL's paired medians and the gate")
    s.add_argument("jsonl", type=Path)
    args = p.parse_args(argv)
    if args.command == "summary":
        return summary(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
