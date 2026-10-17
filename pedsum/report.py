"""Report payload builders, safe-attempt redaction, and output writers."""

from __future__ import annotations

import gzip
import math
import os
import shutil
import subprocess
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Union

import polars as pl
import yaml

from pedsum.base import VERSION, PedigreeError, logger
from pedsum.checks import CHECK_GROUPS, CHECK_LABELS
from pedsum.parse import plink_sex, read_aligned_columns
from pedsum.redaction import SAFE_MIN_CELL
from pedsum.schema import SECTION_SPECS, categorise_pedigree, split_individual_distributions, split_summary
from pedsum.sections import INBREEDING_COLUMNS, Distribution, numeric_distribution

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from pathlib import Path

    from pedsum.checks import CheckResult, Finding
    from pedsum.ids import IdLabels
    from pedsum.validate import ValidationConfig

#: One value in the ``relationship_pairs`` section: a count (``None`` when the
#: code was not computed) or the ``by_degree`` rollup.
PairEntry = Union[int, None, "Mapping[int, int | None]"]

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
    "ecg",
    "n_descendant_paths",
)

# pedsum's own provenance columns that ride along in a re-ingestible
# ``validate.tsv.gz``.  When such a file is later summarized they collide with
# the freshly derived columns; the stale copy is regenerated, so it is dropped
# rather than preserved under an ``_input`` suffix (see write_annotated_tsv).
_RESERVED_PROVENANCE_COLS = frozenset({"sex_source"})


@contextmanager
def atomic_output(path: Path) -> Generator[Path]:
    """Yield a temporary path beside ``path`` and move it onto ``path`` when the block succeeds.

    The temporary is ``.<name>.partial-<pid>`` in the same directory, so the
    final ``os.replace`` stays on one filesystem and a reader never sees a
    half-written output. When the block raises, the temporary is removed and
    the previous ``path``, if any, is left as it was. A process stopped by
    ``os._exit`` mid-write leaves its ``.partial-<pid>`` file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


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


def _meta(path: Path, cmd: str, n_total: int | None) -> dict:
    """The block every report starts with, in this key order: what ran on which input, and how many rows."""
    return {
        "input": str(path),
        "command": cmd,
        "version": VERSION,
        "generated_at": _now_iso(),
        "n_total": n_total,
    }


def build_effective_size_data(
    path: Path,
    cmd: str,
    n_total: int | None,
    status: str,
    requested: list[str],
    effective_size: dict,
) -> dict:
    """The ``effective_size.yaml`` payload: the summary's meta block, then the run status and every estimator."""
    return {
        **_meta(path, cmd, n_total),
        "status": status,
        "estimators_requested": requested,
        "effective_size": effective_size,
    }


#: ``assortative_mating.yaml`` keeps this many significant digits below 0.1, where 4 dp would zero a small
#: p-value, SE or share.
ASSORTATIVE_MATING_FIGURES = 4


def build_assortative_mating_data(path: Path, cmd: str, n_total: int, assortative_mating: dict) -> dict:
    """The ``assortative_mating.yaml`` payload: the summary's meta block, then the Mate Correlation section."""
    return {**_meta(path, cmd, n_total), "assortative_mating": assortative_mating}


def build_pedigree_data(
    path: Path,
    cmd: str,
    size: dict,
    sibships: dict | None,
    pairs: dict,
    inbreeding: dict | None,
    mating_pairs: dict | None,
    relationship_summary: dict | None,
    aggregates: dict,
    sex_concordance: dict | None,
    *,
    max_degree: int = 5,
    multiple_births: dict | None = None,
) -> dict:
    """Build the flat pedigree-level payload.

    ``write_long_tsv`` flattens it into ``summary.pedigree.tsv``, and
    ``build_summary_data`` categorises it into the two summary YAMLs. Each
    section arrives as YAML-ready native types; a None section is absent.
    """
    return {
        **_meta(path, cmd, size["n_total"]),
        "max_degree_enumerated": max_degree,
        "size_structure": _size_structure(size),
        "sibship_size": sibships,
        "mating_pairs": mating_pairs,
        # Opt-in (`--sex-concordance`); None keeps the section out of every
        # output entirely, which is what holds the slim YAML line budget.
        "offspring_sex_concordance": sex_concordance,
        # Opt-in (`--multiple-birth-col`). Absent rather than None, so a run
        # without it writes no empty "meta multiple_births" row to the long TSV.
        **({"multiple_births": multiple_births} if multiple_births is not None else {}),
        "relationship_summary": relationship_summary,
        "reproduction": aggregates["reproduction"],
        "genealogy": aggregates["genealogy"],
        "founder_contribution": aggregates["founder_contribution"],
        "founder_summary": aggregates["founder_summary"],
        "components": aggregates["components"],
        "sex_summary": aggregates["sex_summary"],
        "depth_summary": aggregates["depth_summary"],
        "pairs_engine": str(pairs.get("_engine", "unknown")),
        "relationship_pairs": {k: _serialise_pair_entry(v) for k, v in pairs.items() if not k.startswith("_")},
        "inbreeding": inbreeding,
    }


def _size_structure(size: dict) -> dict:
    """The ``size_structure`` section; ``sex_source`` follows ``n_female`` when the input carried the column."""
    out = {}
    for k in _SIZE_STRUCTURE_KEYS:
        out[k] = size[k]
        if k == "n_female" and size.get("sex_source") is not None:
            out["sex_source"] = size["sex_source"]
    return out


def _serialise_pair_entry(value: PairEntry) -> object:
    """Coerce one ``relationship_pairs`` entry to its output form.

    Two shapes share the section, and each one's type says how to render it:
    the ``by_degree`` rollup is a mapping keyed by degree, and everything else
    is a count. A count is ``None`` when pedigree-graph was not asked for that
    code, which stays ``None`` rather than becoming a misleading 0.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(degree): None if count is None else int(count) for degree, count in value.items()}
    return int(value)


def build_individual_data(
    idf: pl.DataFrame,
    path: Path,
    cmd: str,
) -> dict:
    """Canonical nested dict for the per-individual distribution report."""
    distributions = {col: numeric_distribution(idf[col]) for col in _NUMERIC_COLS if col in idf.columns}
    return {**_meta(path, cmd, len(idf)), "distributions": distributions}


_SUMMARY_META_KEYS = ("input", "command", "version", "generated_at", "n_total")


def apply_safe_attempt(ped_data: dict, ind_data: dict, min_cell: int = SAFE_MIN_CELL) -> None:
    """Best-effort small-cell redaction (in place). Not a safe-harbor.

    Runs each summary section's ``SectionSpec.redact`` rule, removes the
    extrema of every ``Distribution`` in the pedigree payload, and in each
    per-individual distribution also nulls a small nonzero count.
    """
    n_total = int(ped_data["n_total"])
    for spec in SECTION_SPECS:
        value = ped_data.get(spec.name)
        if spec.redact is not None and value is not None:
            spec.redact(value, min_cell, n_total)
    _strip_extrema(ped_data)
    distributions = ind_data["distributions"]
    for col, dist in distributions.items():
        small_nz = 0 < (dist.nz or 0) < min_cell
        distributions[col] = replace(dist.without_extrema(), nz=None if small_nz else dist.nz)


def _strip_extrema(obj: dict | list) -> None:
    """Replace every ``Distribution`` nested in ``obj`` by its copy without extrema."""
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    for key, value in list(items):
        if isinstance(value, Distribution):
            obj[key] = value.without_extrema()
        elif isinstance(value, (dict, list)):
            _strip_extrema(value)


def as_plain(payload: dict) -> dict:
    """``payload`` with each ``Distribution`` rendered as its mapping, for the writers that walk dicts and lists."""
    return {k: _plain(v) for k, v in payload.items()}


def _plain(obj: object) -> object:
    if isinstance(obj, Distribution):
        return obj.as_dict()
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def build_summary_data(ped_data: dict, ind_data: dict) -> tuple[dict, dict]:
    """Build the (slim, extra) categorised YAML payloads from flat dicts.

    Pipeline: strip meta → categorise → split per ``SUMMARY_SCHEMA``.
    The same meta block sits at the top
    of both files so each is self-identifying. Per-individual
    ``distributions`` gets its own slim/extra split via
    ``split_individual_distributions``.

    ``ped_data`` is left untouched (the TSV writers read it directly).
    """
    meta = {k: ped_data[k] for k in _SUMMARY_META_KEYS}

    flat_ped = {k: v for k, v in ped_data.items() if k not in _SUMMARY_META_KEYS}
    nested_ped = categorise_pedigree(flat_ped)
    slim_ped, extra_ped = split_summary(nested_ped)

    ind_payload = {k: v for k, v in ind_data.items() if k not in _SUMMARY_META_KEYS}
    dists = ind_payload.get("distributions", {})
    slim_dists, extra_dists = split_individual_distributions(dists)
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


def _round_floats(obj: object, ndigits: int = 4, figures: int = 0) -> object:
    """Recursively round floats in nested dicts/lists to ``ndigits`` decimals, or to ``figures`` significant digits when that keeps more."""
    if isinstance(obj, float):
        if figures and obj and math.isfinite(obj):
            return round(obj, max(ndigits, figures - 1 - math.floor(math.log10(abs(obj)))))
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits, figures) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(x, ndigits, figures) for x in obj]
    return obj


def write_yaml(data: dict, path: Path, figures: int = 0) -> None:
    """Write data as YAML to path (creates parent dirs); floats rounded as ``_round_floats`` does."""
    with atomic_output(path) as tmp, tmp.open("w") as fh:
        yaml.safe_dump(_round_floats(data, figures=figures), fh, sort_keys=False, default_flow_style=False)


def _render_cell(value: object) -> str | None:
    """One long-TSV cell, rendered exactly as the historical pandas writer did.

    pandas ``to_csv`` applied ``str()`` to every object-column cell and wrote
    ``None`` as an empty field. Empty cells are kept as nulls (not ``""``) so
    the polars CSV writer emits a bare empty field instead of a quoted ``""``.
    """
    if value is None or value == "":
        return None
    return str(value)


def write_long_tsv(data: dict, path: Path) -> None:
    """Write data flattened to a long-form TSV; floats rounded to 4dp."""
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
    with atomic_output(path) as tmp:
        df.write_csv(tmp, separator="\t")


def prepare_out_dir(path: Path) -> int:
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


def to_csv_gz(df: pl.DataFrame, out_path: Path) -> None:
    """Write ``df`` as gzipped TSV; uses pigz when available, else gzip level 1.

    Output is standard .gz either way.
    """
    df = _csv_ready(df)
    pigz = shutil.which("pigz")
    if pigz is not None:
        with (
            atomic_output(out_path) as tmp,
            tmp.open("wb") as fh_out,
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
    with atomic_output(out_path) as tmp, gzip.open(tmp, "wb", compresslevel=1) as fh:
        # GzipFile satisfies the IO[bytes] target at runtime; polars' overloads
        # spell it more narrowly than ty can match.
        df.write_csv(fh, separator="\t")  # ty: ignore[no-matching-overload]


def write_annotated_tsv(
    in_path: Path,
    config: ValidationConfig,
    sep: str,
    idf: pl.DataFrame,
    out_path: Path,
    labels: IdLabels,
) -> None:
    """Re-read input pedigree, append derived columns, write annotated tsv.gz.

    Output preserves input columns under canonical names (id/sex/mother/
    father; user-supplied names are renamed), with IDs as the input wrote
    them. With string IDs every input column is re-read as text, since type
    inference would turn ``001`` into ``1``. Sex is the resolved sex in
    PLINK's coding (1=male, 2=female, 0=unknown). All derived per-individual columns are
    appended. Row order matches input. Gzipped tab-separated.
    """
    raw = read_aligned_columns(
        in_path, sep, config.id_col, idf["id"].to_numpy(), labels, infer_types=not labels.is_string
    )
    if raw[config.id_col].null_count():
        # An idf row the input lacks: rows were added or dropped between the
        # input and idf, not a benign reorder.
        raise PedigreeError("internal: row order mismatch between input and individual table")
    rename_map = {
        config.id_col: "id",
        config.sex_col: "sex",
        config.mother_col: "mother",
        config.father_col: "father",
    }
    rename_map = {k: v for k, v in rename_map.items() if k != v}
    if rename_map:
        raw = raw.rename(rename_map)

    canonical = ("id", "sex", "mother", "father")
    extras = raw.drop([c for c in canonical if c in raw.columns])

    # The inbreeding columns stay reserved under --no-inbreeding, so an input
    # column of that name keeps its _input suffix instead of passing as computed.
    derived_cols = (set(idf.columns) | set(INBREEDING_COLUMNS)) - set(canonical)
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

    idf = idf.with_columns(pl.Series("sex", plink_sex(idf["sex"].to_numpy())))
    idf = labels.relabel(idf, "id", "mother", "father")
    annotated = pl.concat([idf, extras], how="horizontal", strict=True)
    to_csv_gz(annotated, out_path)


def format_check_summary(path: Path, n_total: int, results: list[CheckResult]) -> str:
    """Render the validate summary as grouped sections with friendly labels.

    The internal check names remain in ``.validate.log``; only the on-screen
    rendering changes here. Unknown check names (e.g. after a partial rename)
    are skipped defensively rather than crashing the formatter.
    """
    by_check = {r.name: r for r in results}
    width = max(len(label) for label in CHECK_LABELS.values()) + 1
    lines = [f"pedsum: validating {path} (N={n_total:,})"]
    total_findings = 0
    for group_name, check_names in CHECK_GROUPS:
        lines.append("")
        lines.append(group_name)
        for name in check_names:
            r = by_check.get(name)
            if r is None:
                continue  # defensive: tolerate missing check names
            label = CHECK_LABELS.get(name, name)
            line = f"  {label} {'.' * (width - len(label))} {r.status}"
            if r.status == "FAIL":
                line += f" ({r.count})"
                total_findings += r.count
            elif r.note:
                line += f" ({r.note})"
            lines.append(line)
    lines.append("")
    lines.append(f"result: {total_findings} finding(s)")
    return "\n".join(lines) + "\n"


def write_validate_log(findings: list[Finding], out_path: Path, labels: IdLabels | None) -> None:
    """Tab-separated log: one row per finding (check / id / row / detail / n_refs).

    ``labels`` is None when the id column did not parse, so no finding has an id.
    """
    # Missing id/row are nulls (not "") so the CSV writer emits a bare empty
    # field; polars quotes explicit empty strings as "".
    df = pl.DataFrame(
        {
            "check": [f.check for f in findings],
            "id": [None if f.id is None or labels is None else labels.label(f.id) for f in findings],
            "row": [None if f.row is None else str(f.row) for f in findings],
            "detail": [f.detail for f in findings],
            "n_refs": [f.n_refs for f in findings],
        },
        schema={"check": pl.String, "id": pl.String, "row": pl.String, "detail": pl.String, "n_refs": pl.Int64},
    )
    with atomic_output(out_path) as tmp:
        df.write_csv(tmp, separator="\t")


def write_cleared_links(cleared: list[tuple[str, str, str, str]], out_path: Path) -> None:
    """Tab-separated manifest of the parent links ``validate --break-cycles`` cleared.

    One row per link: the child's id, the parent ``role``, the ``parent`` id
    and the ``reason``. Written even when nothing was cleared (header only).
    """
    df = pl.DataFrame(
        {name: [r[i] for r in cleared] for i, name in enumerate(("id", "role", "parent", "reason"))},
        schema=dict.fromkeys(("id", "role", "parent", "reason"), pl.String),
    )
    with atomic_output(out_path) as tmp:
        df.write_csv(tmp, separator="\t")


def write_dropped_manifest(dropped: list[tuple[str, str, int]], out_path: Path) -> None:
    """Tab-separated manifest of `validate --drop-offending` removals.

    One row per ``(id, check, round)``, which ``reduce_pedigree`` emits distinct: the id removed, the Check that
    flagged it, and the reduction round in which it was dropped (later rounds
    record offenders spawned by earlier drops). Written even when nothing was
    dropped (header only) so downstream tooling can rely on its presence.
    """
    df = pl.DataFrame(
        {
            "id": [r[0] for r in dropped],
            "check": [r[1] for r in dropped],
            "round": [r[2] for r in dropped],
        },
        schema={"id": pl.String, "check": pl.String, "round": pl.Int64},
    )
    with atomic_output(out_path) as tmp:
        df.write_csv(tmp, separator="\t")
