"""Categorised summary YAML schema and slim/extra split machinery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionSpec:
    """One leaf section under a category."""
    name: str
    slim_keys: tuple[str, ...] | None = None
    list_of_dict_slim_keys: tuple[str, ...] | None = None

@dataclass(frozen=True)
class CategorySpec:
    """One category bucket grouping related sections."""
    name: str
    sections: tuple[SectionSpec, ...]

# 23 named relationship codes from REL_REGISTRY plus PO and engine.
_PAIRS_SLIM_KEYS: tuple[str, ...] = (
    "MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "Av", "GGP", "HAv", "GAv",
    "1C", "GGGP", "HGAv", "GGAv", "H1C", "1C1R", "G3GP", "HGGAv", "G3Av",
    "H1C1R", "1C2R", "2C", "PO", "engine",
)

SUMMARY_SCHEMA: tuple[CategorySpec, ...] = (
    CategorySpec("structure", (
        SectionSpec("size_structure"),
        SectionSpec("components"),
        SectionSpec("max_degree_enumerated"),
    )),
    CategorySpec("demography", (
        SectionSpec("sibship_size"),
        SectionSpec("mating_pairs"),
    )),
    CategorySpec("individuals", (
        SectionSpec("reproduction"),
        SectionSpec("genealogy"),
    )),
    CategorySpec("founders", (
        SectionSpec("founder_contribution"),
        SectionSpec("founder_summary"),
    )),
    CategorySpec("relatedness", (
        SectionSpec("relationship_pairs", slim_keys=_PAIRS_SLIM_KEYS),
        SectionSpec("relationship_summary"),
        SectionSpec("inbreeding"),
    )),
    CategorySpec("popgen", (
        SectionSpec("effective_size"),  # per-estimator special handling
    )),
    CategorySpec("strata", (
        SectionSpec("sex_summary"),     # per-stratum special handling
        SectionSpec("depth_summary", list_of_dict_slim_keys=("depth", "n")),
    )),
)

# Per-stratum slim keys for sex_summary (the scalar integers only).
_SEX_SUMMARY_SLIM_KEYS: tuple[str, ...] = (
    "n", "n_founders", "n_reproductive", "n_inbred",
)

# Per-individual distribution split: only headline columns in slim, and within
# each column only mean+median; extra carries all columns × all quantile keys.
INDIVIDUAL_SLIM_COLS: tuple[str, ...] = ("n_offspring", "n_mates", "F", "n_distinct_ancestors")

INDIVIDUAL_SLIM_DIST_KEYS: tuple[str, ...] = ("mean", "median")

# YAML-only drops (paths within the categorised dict). Constructed in the
# flat dict and visible in the TSV; excluded from both slim and extra YAML.
KNOWN_YAML_DROPS: frozenset[str] = frozenset({
    "pedigree.relatedness.relationship_pairs.by_degree",
    "individual.distributions.F.max",
})

def _categorise_pedigree(flat_ped: dict) -> dict:
    """Wrap the flat pedigree dict into the category structure.

    Pure function. Reads ``SUMMARY_SCHEMA`` to bucket sections by
    category, drops empty categories, and folds the sibling
    ``pairs_engine`` into ``relationship_pairs.engine`` so it lives
    inside the relatedness.relationship_pairs subtree.

    Does NOT apply slim/extra splitting — that is ``_split_summary``'s
    job. Does NOT apply ``KNOWN_YAML_DROPS`` — splitter handles them.
    """
    sections = dict(flat_ped)
    pairs = sections.get("relationship_pairs")
    pairs_engine = sections.pop("pairs_engine", None)
    if pairs is not None and pairs_engine is not None:
        # Non-destructive: build a new dict so the caller's flat payload
        # (which the TSV writer also reads) is not mutated.
        sections["relationship_pairs"] = {**pairs, "engine": pairs_engine}

    nested: dict = {}
    for cat in SUMMARY_SCHEMA:
        cat_dict: dict = {}
        for sec in cat.sections:
            value = sections.get(sec.name)
            if value is None:
                continue
            if isinstance(value, dict) and not value:
                continue
            if isinstance(value, list) and not value:
                continue
            cat_dict[sec.name] = value
        if cat_dict:
            nested[cat.name] = cat_dict
    return nested

# Per-estimator routing for effective_size. Detection is name-based (not
# value-based) so a field that *normally* carries an array still routes to
# extra when upstream emits ``None`` (e.g. Hill arrays when birth_year is
# absent and the estimator collapses).
_EFFECTIVE_SIZE_ARRAY_SUFFIXES: tuple[str, ...] = (
    "_per_gen", "_per_transition", "_per_cohort",
)

_EFFECTIVE_SIZE_ARRAY_NAMES: frozenset[str] = frozenset({
    "cohort_years", "age_table",
    "v_mm", "v_mf", "v_fm", "v_ff", "cov_m", "cov_f",
})

def _is_effective_size_array_key(key: str) -> bool:
    """Return True if ``key`` names an array field inside an Ne estimator."""
    if key in _EFFECTIVE_SIZE_ARRAY_NAMES:
        return True
    return any(key.endswith(suf) for suf in _EFFECTIVE_SIZE_ARRAY_SUFFIXES)

def _split_effective_size(es_dict: dict) -> tuple[dict, dict]:
    """Per-estimator scalar/array split for ``popgen.effective_size``.

    For each estimator: scalars and small dicts like ``cohort_window``
    stay in slim; per-generation / per-cohort / per-transition arrays
    plus ``age_table`` go to extra. Routing is name-based, so
    placeholder ``None`` values for unpopulated arrays still land in
    extra. ``ne_coancestry`` with ``ne is None`` gets a slim-only
    ``{ne: null}`` stub and no extra entry.
    """
    slim: dict = {}
    extra: dict = {}
    for est_name, est_value in es_dict.items():
        if not isinstance(est_value, dict):
            slim[est_name] = est_value
            continue
        if est_name == "ne_coancestry" and est_value.get("ne") is None:
            slim[est_name] = {"ne": None}
            continue
        est_slim: dict = {}
        est_extra: dict = {}
        for k, v in est_value.items():
            if _is_effective_size_array_key(k):
                est_extra[k] = v
            else:
                est_slim[k] = v
        if est_slim:
            slim[est_name] = est_slim
        if est_extra:
            extra[est_name] = est_extra
    return slim, extra

def _split_sex_summary(sex_dict: dict) -> tuple[dict, dict]:
    """Per-stratum split for ``strata.sex_summary``: scalars in slim, dists in extra."""
    slim: dict = {}
    extra: dict = {}
    for stratum_name, stratum_value in sex_dict.items():
        if not isinstance(stratum_value, dict):
            slim[stratum_name] = stratum_value
            continue
        s_slim = {k: stratum_value[k] for k in _SEX_SUMMARY_SLIM_KEYS if k in stratum_value}
        s_extra = {k: v for k, v in stratum_value.items() if k not in _SEX_SUMMARY_SLIM_KEYS}
        if s_slim:
            slim[stratum_name] = s_slim
        if s_extra:
            extra[stratum_name] = s_extra
    return slim, extra

def _split_section(value, spec: SectionSpec) -> tuple[object, object | None]:
    """Split a single section value into (slim, extra) per its SectionSpec."""
    if not isinstance(value, (dict, list)):
        return value, None  # scalar section → slim only
    if isinstance(value, list):
        if spec.list_of_dict_slim_keys is None:
            return value, None
        slim_rows: list = []
        extra_rows: list = []
        any_extra = False
        for row in value:
            if not isinstance(row, dict):
                slim_rows.append(row)
                extra_rows.append(None)
                continue
            keep = spec.list_of_dict_slim_keys
            slim_rows.append({k: row[k] for k in keep if k in row})
            row_extra = {k: v for k, v in row.items() if k not in keep}
            extra_rows.append(row_extra if row_extra else {})
            if row_extra:
                any_extra = True
        return slim_rows, (extra_rows if any_extra else None)
    # Dict section.
    if spec.name == "effective_size":
        s, e = _split_effective_size(value)
        return s, (e if e else None)
    if spec.name == "sex_summary":
        s, e = _split_sex_summary(value)
        return s, (e if e else None)
    if spec.slim_keys is None:
        return value, None
    slim_dict = {k: value[k] for k in spec.slim_keys if k in value}
    extra_dict = {k: v for k, v in value.items() if k not in spec.slim_keys}
    return slim_dict, (extra_dict if extra_dict else None)

def _drop_dotted_path(d: dict, parts: tuple[str, ...]) -> None:
    """Delete ``d[parts[0]][parts[1]]...`` if present. In-place; safe on missing keys."""
    if not parts or not isinstance(d, dict):
        return
    if len(parts) == 1:
        d.pop(parts[0], None)
        return
    sub = d.get(parts[0])
    if isinstance(sub, dict):
        _drop_dotted_path(sub, parts[1:])

def _split_summary(nested_ped: dict) -> tuple[dict, dict]:
    """Split a categorised pedigree dict into (slim, extra) per ``SUMMARY_SCHEMA``.

    Implements the split contract: every leaf key in ``nested_ped``
    appears in exactly one of (slim, extra, ``KNOWN_YAML_DROPS``). Empty
    categories are omitted from both files.
    """
    spec_by_name = {sec.name: sec for cat in SUMMARY_SCHEMA for sec in cat.sections}
    slim: dict = {}
    extra: dict = {}
    for cat_name, cat_dict in nested_ped.items():
        slim_cat: dict = {}
        extra_cat: dict = {}
        for sec_name, value in cat_dict.items():
            spec = spec_by_name.get(sec_name)
            if spec is None:
                slim_cat[sec_name] = value
                continue
            slim_val, extra_val = _split_section(value, spec)
            if slim_val not in (None, {}, []):
                slim_cat[sec_name] = slim_val
            if extra_val not in (None, {}, []):
                extra_cat[sec_name] = extra_val
        if slim_cat:
            slim[cat_name] = slim_cat
        if extra_cat:
            extra[cat_name] = extra_cat
    for drop_path in KNOWN_YAML_DROPS:
        if not drop_path.startswith("pedigree."):
            continue
        parts = tuple(drop_path.split(".")[1:])
        _drop_dotted_path(slim, parts)
        _drop_dotted_path(extra, parts)
    return slim, extra

def _split_individual_distributions(
    dists: dict,
) -> tuple[dict, dict]:
    """Split per-individual distributions into (slim, extra) residues.

    For each column:

    * If the column is in ``INDIVIDUAL_SLIM_COLS``, slim keeps the
      ``INDIVIDUAL_SLIM_DIST_KEYS`` (mean/median); extra carries the
      remaining quantile keys (min/std/q1/q3/max/nz).
    * Otherwise the full distribution dict goes to extra; slim has no
      entry for that column.

    Applies the ``individual.distributions.F.max`` YAML-only drop (drops
    from extra; slim never carried ``max`` for ``F`` anyway). Under this
    contract, no leaf path appears in both slim and extra.
    """
    slim_d: dict = {}
    extra_d: dict = {}
    for col, dist in dists.items():
        if not isinstance(dist, dict):
            slim_d[col] = dist
            continue
        if col in INDIVIDUAL_SLIM_COLS:
            slim_d[col] = {k: dist[k] for k in INDIVIDUAL_SLIM_DIST_KEYS if k in dist}
            residue = {k: v for k, v in dist.items() if k not in INDIVIDUAL_SLIM_DIST_KEYS}
            if residue:
                extra_d[col] = residue
        else:
            extra_d[col] = dict(dist)
    if "F" in extra_d and isinstance(extra_d["F"], dict):
        extra_d["F"].pop("max", None)  # KNOWN_YAML_DROPS
    return slim_d, extra_d
