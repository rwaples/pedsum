#!/usr/bin/env python3
"""Median wall and peak RSS of ``pedsum assortative-mating`` over repeated runs.

Each point runs the CLI ``--runs`` times under ``/usr/bin/time -f "%e %M"``
(wall seconds, ``ru_maxrss`` KiB from ``wait4``) and appends one JSON record per
point to ``--out`` (medians plus every run). ``--script`` names the
``pedigree_summary.py`` to run, with its directory as the working directory so
that tree's ``pedsum`` is imported ahead of the editable install (a frozen copy
of the tree can be measured while the working tree changes); ``--python`` is the interpreter (the pixi env's
``python``, so no ``pixi run`` happens inside the timed loop).

Configurations:

* ``a``: one binary trait (``--trait dx``), unstratified.
* ``b``: ``--trait liab dx --stratify-by birth_year`` (all four cell types).

Inputs come from ``generate_assortative_mating.py``. Usage::

    python benchmarks/bench_assortative_mating.py --python .pixi/envs/default/bin/python \
        --script pedigree_summary.py --label before --runs 3 --out /tmp/am_bench.jsonl \
        --point a:/tmp/am_1e4.tsv:999:1000 --point b:/tmp/am_1e4.tsv:999:1000
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CONFIG_ARGS = {
    "a": ["--trait", "dx"],
    "b": ["--trait", "liab", "dx", "--stratify-by", "birth_year", "--birth-year-col", "birth_year"],
}


def run_once(
    python: Path, script: Path, config: str, data: Path, perms: int, boots: int, threads: int
) -> tuple[float, int]:
    """One CLI run; returns (wall seconds, peak RSS KiB)."""
    with tempfile.TemporaryDirectory() as out_dir:
        cmd = [
            "/usr/bin/time",
            "-f",
            "%e %M",
            str(python),
            str(script),
            "assortative-mating",
            "--in",
            str(data),
            "--out",
            out_dir,
            *CONFIG_ARGS[config],
            "--permutations",
            str(perms),
            "--bootstrap",
            str(boots),
            "--threads",
            str(threads),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=script.parent)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"run failed ({proc.returncode}): {' '.join(cmd)}")
    wall, rss = proc.stderr.strip().splitlines()[-1].split()
    return float(wall), int(rss)


def render(path: Path) -> str:
    """The JSONL records as one Markdown table, before and after side by side per (config, pairs, P, B)."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_point: dict[tuple, dict[str, dict]] = {}
    for r in records:
        by_point.setdefault((r["config"], r["pairs"], r["permutations"], r["bootstrap"]), {})[r["label"]] = r
    labels = sorted({r["label"] for r in records})
    head = (
        ["config", "pairs", "P/B"] + [f"{lab} wall s (runs)" for lab in labels] + [f"{lab} RSS MiB" for lab in labels]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in sorted(by_point):
        cells = [key[0], f"{key[1]:,}", f"{key[2]}/{key[3]}"]
        for lab in labels:
            r = by_point[key].get(lab)
            cells.append(
                "" if r is None else f"{r['median_wall_s']:.1f} ({', '.join(f'{x["wall_s"]:.1f}' for x in r['runs'])})"
            )
        for lab in labels:
            r = by_point[key].get(lab)
            cells.append("" if r is None else f"{r['median_max_rss_mb']:.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Time every ``--point`` and append the records to ``--out``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--python", type=Path, help="required unless --render")
    p.add_argument("--script", type=Path, help="required unless --render")
    p.add_argument("--label", default="run", help="e.g. before / after")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--threads", type=int, default=1, help="--threads of every run (default 1)")
    p.add_argument("--out", type=Path, required=True, help="JSONL, appended")
    p.add_argument("--point", action="append", default=[], metavar="CONFIG:DATA:PERMS:BOOTS")
    p.add_argument("--render", action="store_true", help="print --out as a Markdown table and exit")
    args = p.parse_args(argv)
    if args.render:
        print(render(args.out))
        return 0
    if args.python is None or args.script is None:
        p.error("--python and --script are required unless --render")
    for point in args.point:
        config, data, perms, boots = point.rsplit(":", 3)
        data_path = Path(data)
        pairs = json.loads(data_path.with_suffix(data_path.suffix + ".meta.json").read_text())["pairs"]
        runs = []
        for _ in range(args.runs):
            wall, rss = run_once(args.python, args.script, config, data_path, int(perms), int(boots), args.threads)
            runs.append({"wall_s": wall, "max_rss_kb": rss})
            print(
                f"{args.label} {config} pairs={pairs} {perms}/{boots}: {wall:.2f} s, {rss / 1024:.0f} MiB", flush=True
            )
        record = {
            "label": args.label,
            "config": config,
            "pairs": pairs,
            "permutations": int(perms),
            "bootstrap": int(boots),
            "threads": args.threads,
            "runs": runs,
            "median_wall_s": statistics.median(r["wall_s"] for r in runs),
            "median_max_rss_mb": statistics.median(r["max_rss_kb"] for r in runs) / 1024,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with args.out.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
