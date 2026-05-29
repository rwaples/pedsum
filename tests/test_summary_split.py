"""Tests for the slim/extra summary YAML split.

The summary YAML is always split into ``summary.yaml`` (slim, headline
content) and ``summary.extra.yaml`` (per-generation/per-cohort arrays
plus full per-individual quantiles), inside the ``--out`` directory.

These tests pin the split contract: each leaf appears in exactly one of
slim / extra / intentional-drop; per-generation values that look like
duplicates of global scalars are preserved in extra; empty categories
are omitted.

The unit test for ``_split_summary`` imports pedigree_summary as a module
to exercise the splitter on a fabricated list-of-dict section.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
from conftest import EXAMPLE, REPO
from conftest import load_summary_extra_yaml as _load_extra
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run

# Load pedigree_summary.py as a module so we can call its split helpers
# directly. Doing this via importlib avoids requiring an editable install
# of pedsum as a package (which the rest of the suite does not assume).
_spec = importlib.util.spec_from_file_location(
    "pedigree_summary",
    REPO / "pedigree_summary.py",
)
_ps = importlib.util.module_from_spec(_spec)
sys.modules["pedigree_summary"] = _ps
_spec.loader.exec_module(_ps)


# --------------------------------------------------------------------------
# File set & line-budget
# --------------------------------------------------------------------------


def test_extra_yaml_exists_by_default(tmp_path):
    """Both summary YAML files are always written."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    assert (out_dir / "summary.yaml").exists()
    assert (out_dir / "summary.extra.yaml").exists()


def test_slim_yaml_line_budget(tmp_path):
    """Slim YAML on the 200-row example pedigree stays under a small budget.

    Budget: 500 lines. The realistic ~780k-horse pedigree lands around
    500; the example is tiny so it should comfortably fit. Catches
    regressions that re-promote a bulky section into slim.
    """
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    n_lines = len((out_dir / "summary.yaml").read_text().splitlines())
    assert n_lines <= 500, f"slim YAML budget exceeded: {n_lines} lines"


# --------------------------------------------------------------------------
# Schema-driven properties
# --------------------------------------------------------------------------


def test_ne_coancestry_absent_when_not_requested(tmp_path):
    """Without ``--ne-coancestry``, slim has ``{ne: null}`` and extra omits it entirely."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    slim_es = _load_yaml(out_dir)["pedigree"]["popgen"]["effective_size"]
    assert slim_es["ne_coancestry"] == {"ne": None}
    extra_es = _load_extra(out_dir)["pedigree"].get("popgen", {}).get("effective_size", {})
    assert "ne_coancestry" not in extra_es


def test_per_depth_fields_preserved_in_extra(tmp_path):
    """``depth_summary[i]`` per-depth scalars survive — they're not duplicates."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    extra_depth = _load_extra(out_dir)["pedigree"]["strata"]["depth_summary"]
    assert isinstance(extra_depth, list)
    assert len(extra_depth) > 0
    # Per-depth scalars (not duplicates of global inbreeding.mean_F etc.):
    sample = extra_depth[0]
    for key in ("n_male", "n_female", "n_inbred", "mean_F", "max_F"):
        assert key in sample, f"per-depth {key} missing from extra.strata.depth_summary[0]"


def test_empty_categories_omitted(tmp_path):
    """With both opt-outs, popgen and inbreeding are absent (not present as empty)."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--no-inbreeding",
            "--no-effective-size",
        ]
    )
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(out_dir)["pedigree"]
    assert "popgen" not in ped
    # None of the categories that *are* present should be empty dicts.
    for cat_name, cat in ped.items():
        assert cat, f"category {cat_name!r} present but empty"


# --------------------------------------------------------------------------
# Direct unit tests on _split_summary
# --------------------------------------------------------------------------


def test_list_of_dict_split_zips_by_index():
    """A list-of-dict section splits row-by-row into slim / extra residues."""
    nested = {
        "strata": {
            "depth_summary": [
                {"depth": 0, "n": 50, "mean_F": 0.0, "n_inbred": 0},
                {"depth": 1, "n": 75, "mean_F": 0.012, "n_inbred": 5},
            ],
        },
    }
    slim, extra = _ps._split_summary(nested)
    # Slim rows keep only the slim_keys ("depth", "n").
    assert slim["strata"]["depth_summary"] == [
        {"depth": 0, "n": 50},
        {"depth": 1, "n": 75},
    ]
    # Extra rows carry the residue, aligned by index.
    extra_rows = extra["strata"]["depth_summary"]
    assert extra_rows[0] == {"mean_F": 0.0, "n_inbred": 0}
    assert extra_rows[1] == {"mean_F": 0.012, "n_inbred": 5}


def test_schema_no_overlap_between_slim_and_extra(tmp_path):
    """No leaf path lives in both slim and extra (totality / no-duplication).

    Walks every leaf of slim's ``pedigree`` subtree and every leaf of
    extra's ``pedigree`` subtree (likewise for ``individual``), collects
    fully-qualified dotted paths (with ``[i]`` for list indices), and
    asserts the intersection is empty.
    """
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)
    extra = _load_extra(out_dir)

    def _leaf_paths(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from _leaf_paths(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from _leaf_paths(v, f"{prefix}[{i}]")
        else:
            yield prefix

    for top in ("pedigree", "individual"):
        slim_leaves = set(_leaf_paths(slim.get(top, {}), top))
        extra_leaves = set(_leaf_paths(extra.get(top, {}), top))
        overlap = slim_leaves & extra_leaves
        assert not overlap, f"{top}: {len(overlap)} leaf(s) appear in both slim and extra: {sorted(overlap)[:5]}"


def test_known_yaml_drops_absent_from_both_files(tmp_path):
    """``KNOWN_YAML_DROPS`` paths must not appear in slim or extra."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)
    extra = _load_extra(out_dir)

    # relationship_pairs.by_degree must be gone from both files.
    slim_pairs = slim["pedigree"]["relatedness"]["relationship_pairs"]
    extra_pairs = extra["pedigree"].get("relatedness", {}).get("relationship_pairs", {})
    assert "by_degree" not in slim_pairs
    assert "by_degree" not in extra_pairs

    # individual.distributions.F.max must be gone from extra (slim never
    # carried it because the slim-key whitelist for F is mean+median only).
    extra_F = extra["individual"].get("distributions", {}).get("F", {})
    assert "max" not in extra_F


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
