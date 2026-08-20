"""Report payload builders, safe-attempt redaction, and output writers."""

from __future__ import annotations

import gzip
import shutil
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import yaml

from pedsum.base import VERSION, PedigreeError, logger
from pedsum.parse import _as_int_col, _read_pedigree_table
from pedsum.pedigree_ops import IdIndex
from pedsum.schema import _categorise_pedigree, _split_individual_distributions, _split_summary
from pedsum.sections import _numeric_distribution
from pedsum.validate import _CHECK_GROUPS, _CHECK_LABELS

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator
    from pathlib import Path

    from pedsum.checks import CheckResult, Finding

_NUMERIC_COLS = (
    "F",
    "n_full_sibs",
    "n_mat_half_sibs",
    "n_pat_half_sibs",
    "n_offspring",
    "n_mates",
    "n_grandparents",
    "n_grandchildren",
    "n_uncles_aunts",
    "n_first_cousins",
    "n_founder_ancestors",
    "n_distinct_ancestors",
    "n_descendant_paths",
)

# pedsum's own provenance columns that ride along in a re-ingestible
# ``validate.tsv.gz``.  When such a file is later summarized they collide with
# the freshly derived columns; the stale copy is regenerated, so it is dropped
# rather than preserved under an ``_input`` suffix (see _write_annotated_tsv).
_RESERVED_PROVENANCE_COLS = frozenset({"sex_source"})


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Keys projected from ``compute_size_structure``'s output into the
# ``size_structure`` summary section, in emit order. ``n_total`` is lifted to
# the top level and ``n_unknown_sex`` is intentionally not surfaced here.
# ``compute_size_structure`` already returns YAML-clean native types
# (scalars via int()/float(); lists via ndarray.tolist()), so no re-cast.
_SIZE_STRUCTURE_KEYS: tuple[str, ...] = (
    "n_founders",
    "founder_frac",
    "n_nonfounders",
    "nonfounder_frac",
    "n_male",
    "n_female",
    "n_mother_links",
    "n_father_links",
    "n_parent_child_edges",
    "n_with_both_parents",
    "n_with_mother_only",
    "n_with_father_only",
    "n_half_founders",
    "max_depth",
    "mean_depth",
    "median_depth",
    "depth_counts",
    "n_components",
    "largest_component",
    "largest_component_frac",
    "next_components",
)


def _build_pedigree_data(
    path: Path,
    cmd: str,
    size: dict,
    sibships: dict,
    pairs: dict,
    inbreeding: dict | None,
    mating_pairs: dict | None,
    relationship_summary: dict | None,
    aggregates: dict | None = None,
    sex_concordance: dict | None = None,
) -> tuple[dict, dict]:
    """Build the pedigree-level report payloads.

    Returns ``(tsv_payload, yaml_extras)``.

    * ``tsv_payload`` is the dict that gets flattened into
      ``summary.pedigree.tsv`` by ``_write_long_tsv``.  It contains every
      section that should appear as long-form (key, subkey, value) rows.
    * ``yaml_extras`` contains deep structures (e.g. ``effective_size``)
      that should appear in ``summary.yaml`` under ``pedigree:`` but
      should NOT be flattened to TSV.  Currently always returned empty
      from this helper — populated by the caller after Ne computation.
    """
    sibship_section: dict | None
    if sibships.get("empty"):
        sibship_section = None
    else:
        sibship_section = {
            "n_sibships": int(sibships["n_sibships"]),
            "mean": float(sibships["mean"]),
            "median": float(sibships["median"]),
            "q1": float(sibships["q1"]),
            "q3": float(sibships["q3"]),
            "size_dist": {str(k): float(v) for k, v in sibships["size_dist"].items()},
        }
    inb_section: dict | None
    if inbreeding is None:
        inb_section = None
    else:
        inb_section = {
            "n_inbred": int(inbreeding["n_inbred"]),
            "frac_inbred": float(inbreeding["frac_inbred"]),
            "mean_F": float(inbreeding["mean_F"]),
            "max_F": float(inbreeding["max_F"]),
            "hist": {str(k): float(v) for k, v in inbreeding["hist"].items()},
        }
    tsv_payload = {
        "input": str(path),
        "command": cmd,
        "version": VERSION,
        "generated_at": _now_iso(),
        "n_total": int(size["n_total"]),
        "max_degree_enumerated": 5,
        "size_structure": {k: size[k] for k in _SIZE_STRUCTURE_KEYS},
        "sibship_size": sibship_section,
        "mating_pairs": mating_pairs,
        # Opt-in (`--sex-concordance`); None keeps the section out of every
        # output entirely, which is what holds the slim YAML line budget.
        "offspring_sex_concordance": sex_concordance,
        "relationship_summary": relationship_summary,
        "reproduction": (aggregates or {}).get("reproduction", {}),
        "genealogy": (aggregates or {}).get("genealogy", {}),
        "founder_contribution": (aggregates or {}).get("founder_contribution", {}),
        "founder_summary": (aggregates or {}).get("founder_summary", {}),
        "components": (aggregates or {}).get("components", {}),
        "sex_summary": (aggregates or {}).get("sex_summary", {}),
        "depth_summary": (aggregates or {}).get("depth_summary", []),
        "pairs_engine": str(pairs.get("_engine", "matrix")),
        "relationship_pairs": {
            k: ({str(deg): int(c) for deg, c in v.items()} if k == "by_degree" else int(v))
            for k, v in pairs.items()
            if not k.startswith("_")
        },
        "inbreeding": inb_section,
    }
    yaml_extras: dict = {}
    return tsv_payload, yaml_extras


def _build_individual_data(
    idf: pl.DataFrame,
    path: Path,
    cmd: str,
    include_inbreeding: bool,
) -> dict:
    """Canonical nested dict for the per-individual distribution report."""
    n = len(idf)
    distributions: dict[str, dict] = {}
    for col in _NUMERIC_COLS:
        if col in {"F", "n_distinct_ancestors"} and not include_inbreeding:
            continue
        if col not in idf.columns:
            continue
        distributions[col] = _numeric_distribution(idf[col])

    out: dict = {
        "input": str(path),
        "command": cmd,
        "version": VERSION,
        "generated_at": _now_iso(),
        "n_total": n,
        "distributions": distributions,
    }
    return out


_SUMMARY_META_KEYS = ("input", "command", "version", "generated_at", "n_total")

SAFE_MIN_CELL = 5


def _drop_distribution_extrema(obj: object) -> None:
    """Remove min/max from nested distribution dicts for safe-attempt output."""
    if isinstance(obj, dict):
        if {"mean", "q1", "median", "q3", "min", "max"}.issubset(obj.keys()):
            obj.pop("min", None)
            obj.pop("max", None)
        for v in obj.values():
            _drop_distribution_extrema(v)
    elif isinstance(obj, list):
        for v in obj:
            _drop_distribution_extrema(v)


def _null_below(d: dict, keys, min_cell: int) -> None:
    """Null each of ``keys`` in ``d`` whose int value is in ``(0, min_cell)``.

    Missing / ``None`` values collapse to 0 via ``or 0`` and are left
    untouched (the ``0 <`` guard fails), matching the original per-site
    small-cell tests.
    """
    for k in keys:
        if 0 < int(d.get(k, 0) or 0) < min_cell:
            d[k] = None


def _redact_small_group(row: dict, keep: tuple[str, ...], min_cell: int, small_keys) -> None:
    """Small-cell redaction for a grouped row keyed by ``n``.

    If the group's ``n`` is below ``min_cell``, null every key except those
    in ``keep``. Otherwise null only the ``small_keys`` whose own value is
    below ``min_cell`` (via :func:`_null_below`).
    """
    if int(row.get("n", 0)) < min_cell:
        for k in list(row):
            if k not in keep:
                row[k] = None
    else:
        _null_below(row, small_keys, min_cell)


# Offspring Sex Concordance: every key that carries inference or a derived rate
# over the eligible set. Suppressed wholesale when the grouping rests on fewer
# than ``min_cell`` eligible groups; eligibility metadata is retained (and
# small-cell-nulled) so the reader still sees why nothing was reported.
_CONCORDANCE_INFERENCE_KEYS: tuple[str, ...] = (
    "conditioning_male_fraction",
    "n_within_group_pairs",
    "max_group_pair_share",
    "observed_pair_concordance",
    "expected_pair_concordance",
    "excess_concordance",
    "direction",
    "z",
    "p_analytical",
    "p_analytical_holm",
    "p_holm",
    "p_source",
    "all_resolved_excess_concordance",
)

_CONCORDANCE_COUNT_KEYS: tuple[str, ...] = (
    "n_groups_total",
    "n_groups_eligible",
    "n_groups_incomplete",
    "n_groups_too_small",
    "n_offspring_total",
    "n_offspring_eligible",
    "n_offspring_sex_excluded",
    "n_offspring_in_small_groups",
)


def _redact_concordance_block(block: dict, min_cell: int) -> None:
    """Small-cell redaction for one Offspring Sex Concordance grouping.

    Recurses once into the ``all_resolved`` sensitivity block, which carries
    the same eligibility/statistic keys and none of the outer wrappers.
    Permutation count, seed, and backend survive — they describe the procedure,
    not the pedigree.
    """
    if 0 < int(block.get("n_groups_eligible", 0) or 0) < min_cell:
        for key in _CONCORDANCE_INFERENCE_KEYS:
            if key in block:
                block[key] = None
        if "male_proportion_distribution" in block:
            block["male_proportion_distribution"] = None
        if "by_group_size" in block:
            block["by_group_size"] = []
        perms = block.get("permutations")
        if isinstance(perms, dict):
            perms["p_raw"] = None
            perms["p_holm"] = None

    for row in block.get("by_group_size") or []:
        if int(row.get("n_groups", 0)) < min_cell:
            for key in list(row):
                if key not in ("group_size", "n_groups"):
                    row[key] = None
        else:
            _null_below(row, ("n_offspring",), min_cell)

    _null_below(block, _CONCORDANCE_COUNT_KEYS, min_cell)
    by_source = block.get("eligible_by_sex_source")
    if isinstance(by_source, dict):
        _null_below(by_source, list(by_source), min_cell)

    nested = block.get("all_resolved")
    if isinstance(nested, dict):
        _redact_concordance_block(nested, min_cell)


def _apply_safe_attempt(ped_data: dict, ind_data: dict, min_cell: int = SAFE_MIN_CELL) -> None:
    """Best-effort small-cell redaction (in place). Not a safe-harbor.

    - Pedigree-level: nulls ``relationship_pairs`` codes and ``inbreeding``
      fields below ``min_cell``; suppresses ``hist`` buckets whose implied
      count (``frac × n_total``) is below the threshold; drops
      ``next_components`` entries below it; nulls positional ``depth_counts``
      entries below it.
    - Individual-level: drops ``min``/``max`` from every distribution, nulls
      ``nz`` below threshold.
    """
    n_total = int(ped_data.get("n_total", 0))

    sizes = ped_data.get("size_structure", {})
    sizes["next_components"] = [s for s in sizes.get("next_components", []) if s >= min_cell]
    sizes["depth_counts"] = [(g if g >= min_cell else None) for g in sizes.get("depth_counts", [])]
    _null_below(sizes, ("largest_component",), min_cell)

    _drop_distribution_extrema(ped_data)

    sibship = ped_data.get("sibship_size")
    mating = ped_data.get("mating_pairs")
    n_pairs = int(mating.get("n_pairs", 0)) if mating is not None else 0
    if sibship is not None and 0 < n_pairs < min_cell:
        size_dist = sibship.get("size_dist")
        if isinstance(size_dist, dict):
            sibship["size_dist"] = dict.fromkeys(size_dist)

    if mating is not None:
        if 0 < int(mating.get("n_pairs", 0)) < min_cell:
            for k in list(mating):
                if k != "n_pairs":
                    mating[k] = None
        _null_below(mating, ("n_pairs_with_multiple_children",), min_cell)

    concordance = ped_data.get("offspring_sex_concordance")
    if isinstance(concordance, dict):
        for name, block in concordance.items():
            if name != "null_model" and isinstance(block, dict):
                _redact_concordance_block(block, min_cell)

    rel_summary = ped_data.get("relationship_summary") or {}
    _null_below(rel_summary, ("n_related_pairs", "n_unrelated_pairs"), min_cell)
    for section in ("related_pairs_by_closest_degree", "closest_relationship_per_individual"):
        vals = rel_summary.get(section)
        if isinstance(vals, dict):
            _null_below(vals, list(vals), min_cell)
    for row in rel_summary.get("related_pair_density_by_depth", []):
        _redact_small_group(
            row,
            ("depth", "n"),
            min_cell,
            ("n_individual_pairs", "n_related_pairs", "n_unrelated_pairs"),
        )

    reproduction = ped_data.get("reproduction", {})
    _null_below(reproduction, ("n_reproductive", "n_terminal"), min_cell)

    founder = ped_data.get("founder_contribution", {})
    _null_below(
        founder,
        ("n_founders_with_descendants", "n_founders_without_descendants"),
        min_cell,
    )

    founder_summary = ped_data.get("founder_summary", {})
    for row in founder_summary.get("by_depth", []):
        _redact_small_group(row, ("depth", "n"), min_cell, ("active_founders",))
    bottleneck = founder_summary.get("bottleneck")
    if isinstance(bottleneck, dict):
        _null_below(bottleneck, ("min_active_founders",), min_cell)

    comps_full = ped_data.get("components", {})
    _null_below(comps_full, ("singletons",), min_cell)

    sex_summary = ped_data.get("sex_summary", {})
    for stats in sex_summary.values():
        _redact_small_group(
            stats,
            ("n",),
            min_cell,
            ("n_founders", "n_reproductive", "n_terminal", "n_inbred"),
        )

    for row in ped_data.get("depth_summary", []):
        _redact_small_group(
            row,
            ("depth", "n"),
            min_cell,
            ("n_male", "n_female", "n_founders", "n_reproductive", "n_terminal", "n_inbred"),
        )

    pairs = ped_data.get("relationship_pairs", {})
    for code in list(pairs):
        if code == "by_degree":
            _null_below(pairs["by_degree"], list(pairs["by_degree"]), min_cell)
        elif isinstance(pairs[code], int):
            _null_below(pairs, (code,), min_cell)

    inb = ped_data.get("inbreeding")
    if inb is not None:
        if 0 < int(inb.get("n_inbred", 0)) < min_cell:
            inb["n_inbred"] = None
            inb["frac_inbred"] = None
            inb["mean_F"] = None
            inb["max_F"] = None
        for bucket, frac in list(inb.get("hist", {}).items()):
            if frac is None:
                continue
            if 0 < frac * n_total < min_cell:
                inb["hist"][bucket] = None

    for dist in ind_data.get("distributions", {}).values():
        dist.pop("min", None)
        dist.pop("max", None)
        _null_below(dist, ("nz",), min_cell)


def _build_summary_data(
    ped_data: dict,
    ind_data: dict,
    *,
    yaml_extras: dict | None = None,
) -> tuple[dict, dict]:
    """Build the (slim, extra) categorised YAML payloads from flat dicts.

    Pipeline: strip meta → drop ``effective_size_scalars`` (TSV-only)
    → splice ``yaml_extras`` (carries ``effective_size``) → categorise
    → split per ``SUMMARY_SCHEMA``. The same meta block sits at the top
    of both files so each is self-identifying. Per-individual
    ``distributions`` gets its own slim/extra split via
    ``_split_individual_distributions``.

    ``ped_data`` is left untouched (the TSV writers read it directly).
    """
    meta = {k: ped_data[k] for k in _SUMMARY_META_KEYS}

    flat_ped = {k: v for k, v in ped_data.items() if k not in _SUMMARY_META_KEYS}
    # ``effective_size_scalars`` is the TSV's separate scalar projection
    # (built in ``_run_summarize``); it never belonged in YAML. Drop it
    # before categorisation so it doesn't leak into the slim or extra
    # YAML files.
    flat_ped.pop("effective_size_scalars", None)
    if yaml_extras:
        flat_ped.update(yaml_extras)

    nested_ped = _categorise_pedigree(flat_ped)
    slim_ped, extra_ped = _split_summary(nested_ped)

    ind_payload = {k: v for k, v in ind_data.items() if k not in _SUMMARY_META_KEYS}
    dists = ind_payload.get("distributions", {})
    slim_dists, extra_dists = _split_individual_distributions(dists)
    base_ind = {k: v for k, v in ind_payload.items() if k != "distributions"}
    slim_ind = dict(base_ind)
    extra_ind = dict(base_ind)
    if slim_dists:
        slim_ind["distributions"] = slim_dists
    if extra_dists:
        extra_ind["distributions"] = extra_dists

    slim_yaml = {**meta, "pedigree": slim_ped, "individual": slim_ind}
    extra_yaml = {**meta, "pedigree": extra_ped, "individual": extra_ind}
    return slim_yaml, extra_yaml


def _flatten_long(
    obj: object,
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[str, str, str, object]]:
    """Yield (section, key, subkey, value) rows from a nested dict / list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_long(v, (*prefix, str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten_long(v, (*prefix, str(i)))
    else:
        if not prefix:
            return
        section = "meta" if len(prefix) == 1 else prefix[0]
        key = prefix[0] if len(prefix) == 1 else prefix[1]
        subkey = ".".join(prefix[2:]) if len(prefix) > 2 else ""
        yield section, key, subkey, obj


def _round_floats(obj: object, ndigits: int = 4) -> object:
    """Recursively round floats in nested dicts/lists to ``ndigits``."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(x, ndigits) for x in obj]
    return obj


def _write_yaml(data: dict, path: Path) -> None:
    """Write data as YAML to path (creates parent dirs); floats rounded to 4dp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(_round_floats(data), fh, sort_keys=False, default_flow_style=False)


def _render_cell(value: object) -> str | None:
    """One long-TSV cell, rendered exactly as the historical pandas writer did.

    pandas ``to_csv`` applied ``str()`` to every object-column cell and wrote
    ``None`` as an empty field. Empty cells are kept as nulls (not ``""``) so
    the polars CSV writer emits a bare empty field instead of a quoted ``""``.
    """
    if value is None or value == "":
        return None
    return str(value)


def _write_long_tsv(data: dict, path: Path) -> None:
    """Write data flattened to a long-form TSV; floats rounded to 4dp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(_flatten_long(_round_floats(data)))
    df = pl.DataFrame(
        {
            "section": [r[0] for r in rows],
            "key": [r[1] for r in rows],
            "subkey": [_render_cell(r[2]) for r in rows],
            "value": [_render_cell(r[3]) for r in rows],
        },
        schema={"section": pl.String, "key": pl.String, "subkey": pl.String, "value": pl.String},
    )
    df.write_csv(path, separator="\t")


def _prepare_out_dir(path: Path) -> int:
    """Create ``path`` as a directory if needed; refuse if it is a file.

    Returns 0 on success, 1 if the path exists as a non-directory.
    """
    if path.exists() and not path.is_dir():
        logger.error(
            "--out %s exists and is not a directory; pass a directory path",
            path,
        )
        return 1
    path.mkdir(parents=True, exist_ok=True)
    return 0


def _csv_ready(df: pl.DataFrame) -> pl.DataFrame:
    """Pre-format columns whose polars CSV rendering differs from pandas'.

    Boolean columns render as ``True`` / ``False`` (polars writes lowercase);
    nulls stay null (written as empty fields either way).
    """
    bool_cols = [name for name, dtype in df.schema.items() if dtype == pl.Boolean]
    if not bool_cols:
        return df
    return df.with_columns(
        pl.when(pl.col(c).is_null()).then(None).when(pl.col(c)).then(pl.lit("True")).otherwise(pl.lit("False")).alias(c)
        for c in bool_cols
    )


def _to_csv_gz(df: pl.DataFrame, out_path: Path) -> None:
    """Write ``df`` as gzipped TSV; uses pigz when available, else gzip level 1.

    Output is standard .gz either way.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = _csv_ready(df)
    pigz = shutil.which("pigz")
    if pigz is not None:
        with (
            out_path.open("wb") as fh_out,
            subprocess.Popen(
                [pigz, "-1", "-p", "4", "-c"],
                stdin=subprocess.PIPE,
                stdout=fh_out,
            ) as proc,
        ):
            assert proc.stdin is not None  # opened with stdin=PIPE above
            df.write_csv(proc.stdin, separator="\t")
            proc.stdin.close()
            if proc.wait() != 0:
                raise PedigreeError(f"pigz exited with status {proc.returncode}")
        return
    with gzip.open(out_path, "wb", compresslevel=1) as fh:
        df.write_csv(fh, separator="\t")  # ty: ignore[invalid-argument-type]


def _write_annotated_tsv(
    in_path: Path,
    args: argparse.Namespace,
    idf: pl.DataFrame,
    out_path: Path,
) -> None:
    """Re-read input pedigree, append derived columns, write annotated tsv.gz.

    Output preserves input columns under canonical names (id/sex/mother/
    father; user-supplied names are renamed). Sex is the validated int
    encoding (0=female, 1=male). All derived per-individual columns are
    appended. Row order matches input. Gzipped tab-separated.
    """
    raw = _read_pedigree_table(in_path, sep=getattr(args, "sep", "auto"))
    rename_map = {
        args.id_col: "id",
        args.sex_col: "sex",
        args.mother_col: "mother",
        args.father_col: "father",
    }
    rename_map = {k: v for k, v in rename_map.items() if k != v}
    if rename_map:
        raw = raw.rename(rename_map)

    raw_ids = _as_int_col(raw["id"], "id")
    idf_ids = idf["id"].to_numpy()
    if not np.array_equal(raw_ids, idf_ids):
        # ``load_and_validate`` may have reordered rows into topological
        # order when the input was not already sorted parents-before-
        # children.  Realign the raw read to ``idf`` order by ID.
        perm = IdIndex(raw_ids).get_indexer(idf_ids)
        if (perm == -1).any() or len(perm) != len(idf_ids):
            # Truly mismatched (rows added or dropped between input and
            # idf) — not a benign reorder.
            raise PedigreeError("internal: row order mismatch between input and individual table")
        raw = raw[perm]

    canonical = ("id", "sex", "mother", "father")
    extras = raw.drop([c for c in canonical if c in raw.columns])

    derived_cols = set(idf.columns) - set(canonical)
    collisions = [c for c in extras.columns if c in derived_cols]

    # A re-ingested pedsum output carries its own provenance columns (e.g.
    # ``sex_source`` from validate.tsv.gz).  Drop the stale copy — it is
    # regenerated below — instead of parking it under an ``_input`` suffix.
    reserved = [c for c in collisions if c in _RESERVED_PROVENANCE_COLS]
    if reserved:
        extras = extras.drop(reserved)
        logger.info("regenerated pedsum provenance column(s) %s; input copy dropped", reserved)

    user_collisions = [c for c in collisions if c not in _RESERVED_PROVENANCE_COLS]
    if user_collisions:
        new_names = {c: f"{c}_input" for c in user_collisions}
        extras = extras.rename(new_names)
        logger.warning(
            "input columns %s collide with derived columns; preserved as %s",
            user_collisions,
            [new_names[c] for c in user_collisions],
        )

    annotated = pl.concat([idf, extras], how="horizontal")
    _to_csv_gz(annotated, out_path)


def _format_check_summary(path: Path, n_total: int, results: list[CheckResult]) -> str:
    """Render the validate summary as grouped sections with friendly labels.

    The internal check names remain in ``.validate.log``; only the on-screen
    rendering changes here. Unknown check names (e.g. after a partial rename)
    are skipped defensively rather than crashing the formatter.
    """
    by_check = {r.name: r for r in results}
    width = max(len(label) for label in _CHECK_LABELS.values()) + 1
    lines = [f"pedigree_summary.py: validating {path} (N={n_total:,})"]
    total_findings = 0
    for group_name, check_names in _CHECK_GROUPS:
        lines.append("")
        lines.append(group_name)
        for name in check_names:
            r = by_check.get(name)
            if r is None:
                continue  # defensive: tolerate missing check names
            label = _CHECK_LABELS.get(name, name)
            line = f"  {label} {'.' * (width - len(label))} {r.status}"
            if r.status == "FAIL":
                line += f" ({r.count})"
                total_findings += r.count
            elif r.skip_reason and r.status in ("SKIP", "PASS"):
                line += f" ({r.skip_reason})"
            lines.append(line)
    lines.append("")
    lines.append(f"result: {total_findings} finding(s)")
    return "\n".join(lines) + "\n"


def _write_validate_log(findings: list[Finding], out_path: Path) -> None:
    """Tab-separated log: one row per finding (check / id / row / detail)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Missing id/row are nulls (not "") so the CSV writer emits a bare empty
    # field; polars quotes explicit empty strings as "".
    df = pl.DataFrame(
        {
            "check": [f.check for f in findings],
            "id": [None if f.id is None else str(f.id) for f in findings],
            "row": [None if f.row is None else str(f.row) for f in findings],
            "detail": [f.detail for f in findings],
        },
        schema={"check": pl.String, "id": pl.String, "row": pl.String, "detail": pl.String},
    )
    df.write_csv(out_path, separator="\t")


def _write_dropped_manifest(dropped: list[tuple[int, str, int]], out_path: Path) -> None:
    """Tab-separated manifest of `validate --drop-offending` removals.

    One row per distinct ``(id, check, round)`` — the id removed, the Check that
    flagged it, and the reduction round in which it was dropped (later rounds
    record offenders spawned by earlier drops). Written even when nothing was
    dropped (header only) so downstream tooling can rely on its presence.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, str, int]] = set()
    rows: list[tuple[int, str, int]] = []
    for fid, check, rnd in dropped:
        if (fid, check, rnd) not in seen:
            seen.add((fid, check, rnd))
            rows.append((fid, check, rnd))
    df = pl.DataFrame(
        {
            "id": [r[0] for r in rows],
            "check": [r[1] for r in rows],
            "round": [r[2] for r in rows],
        },
        schema={"id": pl.Int64, "check": pl.String, "round": pl.Int64},
    )
    df.write_csv(out_path, separator="\t")


def _build_added_founders(
    mothers: np.ndarray,
    fathers: np.ndarray,
    id_index: IdIndex,
    no_sex_check: bool,
) -> list[dict]:
    """Synthesize founder rows for missing parent IDs, sorted by ID."""
    moms_ref = np.unique(mothers[mothers != -1])
    dads_ref = np.unique(fathers[fathers != -1])
    moms_missing = set(moms_ref[id_index.get_indexer(moms_ref) == -1].tolist())
    dads_missing = set(dads_ref[id_index.get_indexer(dads_ref) == -1].tolist())
    conflicts = moms_missing & dads_missing
    moms_only = moms_missing - conflicts
    dads_only = dads_missing - conflicts

    def _rows_listing(rows: np.ndarray) -> str:
        if len(rows) <= 5:
            return f"row(s) {rows.tolist()}"
        return f"row(s) {rows[:5].tolist()} (and {len(rows) - 5} more)"

    out: list[dict] = []
    for mid in sorted(moms_only):
        rows = np.where(mothers == mid)[0]
        out.append({"id": int(mid), "sex": "F", "reason": f"referenced as mother in {_rows_listing(rows)}"})
    for did in sorted(dads_only):
        rows = np.where(fathers == did)[0]
        out.append({"id": int(did), "sex": "M", "reason": f"referenced as father in {_rows_listing(rows)}"})
    if no_sex_check:
        for cid in sorted(conflicts):
            rows_m = np.where(mothers == cid)[0]
            rows_f = np.where(fathers == cid)[0]
            out.append(
                {
                    "id": int(cid),
                    "sex": "F",
                    "reason": (
                        f"--no-sex-check; conflicting roles "
                        f"(mother {_rows_listing(rows_m)}, father {_rows_listing(rows_f)})"
                    ),
                }
            )
    out.sort(key=lambda x: x["id"])
    return out


def _write_validate_tsv_gz(
    df_raw: pl.DataFrame,
    added_founders: list[dict],
    id_col: str,
    sex_col: str,
    mother_col: str,
    father_col: str,
    out_path: Path,
) -> None:
    """Write input pedigree (gzipped TSV), with new founder rows prepended at top."""
    if added_founders:
        n_new = len(added_founders)
        # Filler cells are nulls (not "") so the CSV writer emits bare empty
        # fields; polars quotes explicit empty strings as "".
        data: dict[str, list[str | None]] = {col: [None] * n_new for col in df_raw.columns}
        data[id_col] = [str(f["id"]) for f in added_founders]
        data[sex_col] = [f["sex"] for f in added_founders]
        data[mother_col] = ["-1"] * n_new
        data[father_col] = ["-1"] * n_new
        new_rows = pl.DataFrame(data, schema=dict.fromkeys(df_raw.columns, pl.String))
        out_df = pl.concat([new_rows, df_raw.cast(pl.String)])
    else:
        out_df = df_raw
    _to_csv_gz(out_df, out_path)
