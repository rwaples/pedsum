"""Check pedsum against statistics published for public pedigrees.

Downloads each pedigree in ``cases.yaml`` from a pinned commit of the CRAN
mirror on GitHub, verifies its SHA-256, converts it to pedsum's input format,
runs ``pedsum validate`` and ``pedsum summarize``, and compares the output with
the published values. Statistics the pedsum CLI does not expose yet are
computed with the ``pedigree_graph`` API that pedsum depends on and labelled
``engine``; statistics neither reports are listed as ``GAP``.

Usage (from the repo root)::

    python benchmarks/published/run_published.py
    python benchmarks/published/run_published.py --only atlas,genea140

Needs network access to raw.githubusercontent.com and the ``rdata`` package
(``pip install rdata``) to read R ``.rda`` files. Writes ``results.md`` and
``results.tsv`` to ``--out`` and exits 1 if any check FAILs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PEDSUM = REPO / "pedigree_summary.py"


@dataclass
class Result:
    """One evaluated check."""

    dataset: str
    stat: str
    published: float
    observed: float | None
    status: str
    via: str
    gap: str
    source: str
    note: str


# ---------------------------------------------------------------------------
# Fetch and convert
# ---------------------------------------------------------------------------


def fetch(url: str, sha256: str, cache: Path) -> Path:
    """Download ``url`` into ``cache`` once and verify its SHA-256."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{sha256[:12]}-{url.rsplit('/', 1)[-1]}"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        path.unlink()
        raise RuntimeError(f"{url}: SHA-256 {digest} does not match the pinned {sha256}")
    return path


def load_table(path: Path, spec: dict[str, Any]) -> dict[str, list[str | None]]:
    """Read the source table into string columns (None for R NA)."""
    if spec.get("format") == "whitespace_table":
        with gzip.open(path, "rt") as fh:
            rows = [line.split() for line in fh if line.strip()]
        header, body = rows[0], rows[1:]
        cols = {name: [r[i] for r in body] for i, name in enumerate(header)}
        return {k: [None if v == "NA" else v for v in vals] for k, vals in cols.items()}
    import rdata  # optional dependency; only the .rda inputs need it

    frame = rdata.read_rda(path)[spec["object"]]
    out: dict[str, list[str | None]] = {}
    for name in frame.columns:
        values = frame[name].astype(object).tolist()
        out[str(name)] = [None if _is_na(v) else _as_token(v) for v in values]
    return out


def _is_na(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)  # NaN and pandas NA compare unequal to themselves
    except TypeError:
        return True  # pandas NA raises on bool()


def _as_token(value: object) -> str:
    """Render R integers stored as float (``12.0``) the way R prints them (``12``)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def convert(table: dict[str, list[str | None]], spec: dict[str, Any], out: Path) -> dict[str, int]:
    """Write pedsum's input TSV and return the original-id -> integer-id map.

    Ids are numbered in first-appearance order over the id column, then over
    parents that have no row of their own (``validate`` adds those as founders).
    Every token in ``missing_parent``, and R NA, means "no parent".
    """
    cols = spec["columns"]
    missing = set(spec.get("missing_parent", []))
    ids = table[cols["id"]]
    mothers = [None if v in missing else v for v in table[cols["mother"]]]
    fathers = [None if v in missing else v for v in table[cols["father"]]]

    id_map: dict[str, int] = {}
    for token in [*ids, *mothers, *fathers]:
        if token is not None and token not in id_map:
            id_map[token] = len(id_map) + 1

    sex_col = cols.get("sex")
    sex_map = {str(k): v for k, v in spec.get("sex_map", {}).items()}
    sexes = table[sex_col] if sex_col else [None] * len(ids)
    ref_col = spec.get("reference_column")
    refs = table[ref_col] if ref_col else [None] * len(ids)

    with out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "sex", "mother", "father", "orig_id", "reference"])
        for i, token in enumerate(ids):
            if token is None:
                raise ValueError(f"row {i}: missing id")
            sex = sex_map.get(sexes[i], "") if sexes[i] is not None else ""
            ref = 1 if refs[i] in ("1", "1.0") else 0
            w.writerow(
                [
                    id_map[token],
                    sex,
                    -1 if mothers[i] is None else id_map[mothers[i]],
                    -1 if fathers[i] is None else id_map[fathers[i]],
                    token,
                    ref,
                ]
            )
    return id_map


# ---------------------------------------------------------------------------
# Run pedsum
# ---------------------------------------------------------------------------


def run_pedsum(args: list[str], log: Path) -> int:
    """Run one pedsum command, appending its output to ``log``."""
    proc = subprocess.run([sys.executable, str(PEDSUM), *args], capture_output=True, text=True, check=False)
    with log.open("a") as fh:
        fh.write(f"$ pedsum {' '.join(args)}\n{proc.stdout}{proc.stderr}\n[exit {proc.returncode}]\n\n")
    return proc.returncode


# ---------------------------------------------------------------------------
# Observed values
# ---------------------------------------------------------------------------


class Observed:
    """Lazily computed observed values for one dataset."""

    def __init__(self, summary: dict[str, Any], annotated: pl.DataFrame, id_map: dict[str, int]) -> None:
        """Hold one dataset's pedsum outputs and its id map."""
        self.summary = summary
        self.annotated = annotated
        self.id_map = id_map
        self._pg: Any = None
        self._ecg: np.ndarray | None = None

    def summary_value(self, path: str) -> float:
        """Scalar at a dotted path of summary.yaml."""
        node: Any = self.summary
        for key in path.split("."):
            node = node[key]
        return float(node)

    def annotated_value(self, kind: str, orig_id: str | None) -> float:
        """Aggregate of the F column of annotated.tsv.gz."""
        f = self.annotated["F"]
        if kind == "sum_F":
            return float(f.sum())
        if kind == "max_F":
            return float(f.max())
        if kind == "F_of":
            return float(self._row_value(f.to_numpy(), self.annotated["id"].to_numpy(), orig_id))
        raise ValueError(f"unknown annotated aggregate {kind!r}")

    def _row_value(self, values: np.ndarray, row_ids: np.ndarray, orig_id: str | None) -> float:
        if orig_id is None:
            raise ValueError("this check needs an `id`")
        hits = np.flatnonzero(row_ids == self.id_map[orig_id])
        if hits.size != 1:
            raise ValueError(f"id {orig_id!r} matched {hits.size} rows")
        return float(values[hits[0]])

    def _graph(self) -> Any:
        if self._pg is None:
            from pedigree_graph import PedigreeGraph

            a = self.annotated
            self._pg = PedigreeGraph.from_arrays(
                ids=a["id"].to_numpy(),
                mother_ids=a["mother"].to_numpy(),
                father_ids=a["father"].to_numpy(),
            )
        return self._pg

    def engine_value(self, name: str, orig_id: str | None) -> float:
        """Statistic computed with the pedigree_graph API (not exposed by the pedsum CLI)."""
        from pedigree_graph.effective_size import ne_individual_delta_f

        pg = self._graph()
        if name.startswith("ecg_"):
            if self._ecg is None:
                # Private helper: pedigree_graph uses it for Ne_iΔF but has no public accessor yet.
                from pedigree_graph._ne_rates import _equivalent_generations

                self._ecg = np.asarray(_equivalent_generations(pg))
            if name == "ecg_sum":
                return float(self._ecg.sum())
            if name == "ecg_max":
                return float(self._ecg.max())
            if name == "ecg_of":
                return self._row_value(self._ecg, self.annotated["id"].to_numpy(), orig_id)
        if name == "ne_idf_t1_reference":
            ref = np.flatnonzero(self.annotated["reference"].fill_null(0).to_numpy() == 1)
            return float(ne_individual_delta_f(pg, reference=ref).ne_unrelated_founders)
        if name == "ne_idf_t1_all":
            ref = np.arange(len(self.annotated))
            return float(ne_individual_delta_f(pg, reference=ref).ne_unrelated_founders)
        raise ValueError(f"unknown engine statistic {name!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate(check: dict[str, Any], obs: Observed | None, error: str | None) -> Result:
    """Compare one published value with pedsum's."""
    via = check["via"]
    published = float(check["published"])
    base = dict(
        dataset=check["dataset"],
        stat=check["stat"],
        published=published,
        via=via,
        gap=check.get("gap", ""),
        source=str(check.get("source", "")),
        note=check.get("note", ""),
    )
    if via == "gap":
        return Result(observed=None, status="GAP", **base)
    if obs is None:
        return Result(observed=None, status="ERROR", **{**base, "note": error or ""})
    kind, _, arg = via.partition(":")
    orig_id = check.get("id")
    try:
        if kind == "summary":
            observed = obs.summary_value(arg)
        elif kind == "annotated":
            observed = obs.annotated_value(arg, orig_id)
        elif kind == "engine":
            observed = obs.engine_value(arg, orig_id)
        else:
            raise ValueError(f"unknown via {via!r}")
    except Exception as exc:  # report and keep going
        return Result(observed=None, status="ERROR", **{**base, "note": f"{type(exc).__name__}: {exc}"})
    observed += check.get("offset", 0)
    tol = float(check.get("tol", 0))
    if math.isclose(observed, published, rel_tol=0, abs_tol=tol + 1e-12):
        status = "PASS"
    elif check.get("status_if_differs") == "expected_difference":
        status = "DIFF"
    else:
        status = "FAIL"
    if kind == "engine" and status == "PASS":
        status = "PASS (engine)"
    return Result(observed=observed, status=status, **base)


def run_dataset(name: str, spec: dict[str, Any], cache: Path, work: Path) -> tuple[Observed | None, str | None]:
    """Fetch, convert and summarize one dataset; return its observed values or an error."""
    ds = work / name
    ds.mkdir(parents=True, exist_ok=True)
    log = ds / "pedsum.log"
    log.write_text("")
    path = fetch(spec["url"], spec["sha256"], cache)
    id_map = convert(load_table(path, spec), spec, ds / "input.tsv")

    extra = ["--allow-missing-sex"] if spec.get("allow_missing_sex") else []
    run_pedsum(["validate", "--in", str(ds / "input.tsv"), "--out", str(ds / "validate"), *extra], log)
    fixed = ds / "validate" / "validate.tsv.gz"
    if not fixed.exists():
        return None, f"validate produced no fixed pedigree; see {log}"
    code = run_pedsum(["summarize", "--in", str(fixed), "--out", str(ds / "summary"), *extra], log)
    if code != 0:
        return None, f"summarize exited {code}; see {log}"
    summary = yaml.safe_load((ds / "summary" / "summary.yaml").read_text())
    annotated = pl.read_csv(ds / "summary" / "annotated.tsv.gz", separator="\t", infer_schema_length=None)
    return Observed(summary, annotated, id_map), None


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.7g}"


def write_reports(results: list[Result], out: Path) -> None:
    """Write results.tsv and results.md."""
    out.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "stat", "published", "observed", "status", "via", "gap", "source", "note"]
    with (out / "results.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(fields)
        for r in results:
            w.writerow(
                [_fmt(v) if k in ("published", "observed") else v for k, v in ((f, getattr(r, f)) for f in fields)]
            )
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    lines = [
        "# pedsum against published pedigree statistics",
        "",
        "Generated by `benchmarks/published/run_published.py`. Statuses: "
        "PASS (pedsum CLI output matches), PASS (engine) (matches, but computed with the "
        "pedigree_graph API because pedsum does not report it yet), DIFF (known convention "
        "difference), GAP (not reported; the gap id refers to docs/missing-statistics.md), "
        "FAIL, ERROR.",
        "",
        "Totals: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "| Dataset | Statistic | Published | pedsum | Status | Gap | Source | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.dataset} | {r.stat} | {_fmt(r.published)} | {_fmt(r.observed)} | {r.status} | {r.gap} | {r.source} | {r.note} |"
        for r in results
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", type=Path, default=HERE / "cases.yaml")
    p.add_argument("--cache", type=Path, default=HERE / ".cache", help="downloaded source files")
    p.add_argument("--work", type=Path, default=HERE / ".work", help="converted inputs and pedsum outputs")
    p.add_argument("--out", type=Path, default=HERE / "results", help="where results.md/.tsv are written")
    p.add_argument("--only", default="", help="comma-separated dataset names")
    args = p.parse_args(argv)

    cases = yaml.safe_load(args.cases.read_text())
    wanted = [s for s in args.only.split(",") if s] or list(cases["datasets"])
    results: list[Result] = []
    for name in wanted:
        print(f"[{name}] running", file=sys.stderr)
        try:
            obs, error = run_dataset(name, cases["datasets"][name], args.cache, args.work)
        except Exception as exc:  # one broken dataset should not hide the others
            obs, error = None, f"{type(exc).__name__}: {exc}"
        results += [evaluate(c, obs, error) for c in cases["checks"] if c["dataset"] == name]

    write_reports(results, args.out)
    width = max(len(r.stat) for r in results)
    for r in results:
        print(
            f"{r.status:14} {r.dataset:9} {r.stat:{width}}  published={_fmt(r.published):>10}  pedsum={_fmt(r.observed):>10}"
        )
    print(f"wrote {args.out / 'results.md'}", file=sys.stderr)
    return 1 if any(r.status in ("FAIL", "ERROR") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
