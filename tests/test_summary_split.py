"""Tests for the slim/extra summary YAML split.

The summary YAML is split into ``<base>.summary.yaml`` (slim, headline
content) and ``<base>.summary.extra.yaml`` (per-generation/per-cohort
arrays plus full per-individual quantiles). ``--single-file`` collapses
the two back into one YAML, still in the categorised structure.

These tests pin the split contract: each leaf appears in exactly one of
slim / extra / intentional-drop; per-generation values that look like
duplicates of global scalars are preserved in extra; the ``--single-file``
mode deletes any pre-existing ``.summary.extra.yaml``; empty categories
(e.g. ``popgen`` without ``--effective-size``) are omitted.

The unit test for ``_split_summary`` + ``_deep_merge_summary`` imports
pedigree_summary as a module to exercise the splitter on a fabricated
list-of-dict section.
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
    "pedigree_summary", REPO / "pedigree_summary.py",
)
_ps = importlib.util.module_from_spec(_spec)
sys.modules["pedigree_summary"] = _ps
_spec.loader.exec_module(_ps)


# --------------------------------------------------------------------------
# File set & line-budget
# --------------------------------------------------------------------------


def test_extra_yaml_exists_by_default(tmp_path):
    """Without ``--single-file``, both summary YAML files are written."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    slim_path = base.parent / f"{base.name}.summary.yaml"
    extra_path = base.parent / f"{base.name}.summary.extra.yaml"
    assert slim_path.exists()
    assert extra_path.exists()


def test_slim_yaml_line_budget(tmp_path):
    """Slim YAML on the 200-row example pedigree stays under a small budget.

    Budget: 500 lines. The realistic ~780k-horse pedigree lands around
    500; the example is tiny so it should comfortably fit. Catches
    regressions that re-promote a bulky section into slim.
    """
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    slim_path = base.parent / f"{base.name}.summary.yaml"
    n_lines = len(slim_path.read_text().splitlines())
    assert n_lines <= 500, f"slim YAML budget exceeded: {n_lines} lines"


# --------------------------------------------------------------------------
# --single-file behaviour
# --------------------------------------------------------------------------


def test_single_file_combines_both(tmp_path):
    """``--single-file`` writes one YAML whose payload equals the in-memory merge."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size", "--single-file",
    ])
    assert res.returncode == 0, res.stderr

    single_path = base.parent / f"{base.name}.summary.yaml"
    extra_path = base.parent / f"{base.name}.summary.extra.yaml"
    assert single_path.exists()
    assert not extra_path.exists()

    # Re-run *without* --single-file at a new basename so we can read both
    # slim and extra from the same source data and merge them in Python;
    # compare the pedigree + individual subtrees (top-level meta will
    # contain a different command string and generated_at timestamp).
    base2 = tmp_path / "q"
    res2 = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base2),
        "--inbreeding", "--effective-size",
    ])
    assert res2.returncode == 0, res2.stderr
    slim = _load_yaml(base2)
    extra = _load_extra(base2)
    merged = _ps._deep_merge_summary(slim, extra)

    single = _load_yaml(base)
    for top_key in ("pedigree", "individual"):
        assert single[top_key] == merged[top_key], f"{top_key} differs"


def test_single_file_removes_stale_extra(tmp_path):
    """Running with ``--single-file`` deletes any pre-existing extra file."""
    base = tmp_path / "p"
    # First run: split mode — creates the extra file.
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    extra_path = base.parent / f"{base.name}.summary.extra.yaml"
    assert extra_path.exists(), "split run should create the extra file"

    # Second run: single-file mode — must delete the stale extra file.
    res2 = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size", "--single-file",
    ])
    assert res2.returncode == 0, res2.stderr
    assert not extra_path.exists(), "stale extra file was not deleted"


# --------------------------------------------------------------------------
# Schema-driven properties
# --------------------------------------------------------------------------


def test_ne_coancestry_absent_when_not_requested(tmp_path):
    """Without ``--ne-coancestry``, slim has ``{ne: null}`` and extra omits it entirely."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    slim_es = _load_yaml(base)["pedigree"]["popgen"]["effective_size"]
    assert slim_es["ne_coancestry"] == {"ne": None}
    extra_es = _load_extra(base)["pedigree"].get("popgen", {}).get("effective_size", {})
    assert "ne_coancestry" not in extra_es


def test_per_generation_fields_preserved_in_extra(tmp_path):
    """``generation_summary[i]`` per-gen scalars survive — they're not duplicates."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    extra_gen = _load_extra(base)["pedigree"]["strata"]["generation_summary"]
    assert isinstance(extra_gen, list)
    assert len(extra_gen) > 0
    # Per-generation scalars (not duplicates of global inbreeding.mean_F etc.):
    sample = extra_gen[0]
    for key in ("n_male", "n_female", "n_inbred", "mean_F", "max_F"):
        assert key in sample, f"per-gen {key} missing from extra.strata.generation_summary[0]"


def test_empty_categories_omitted(tmp_path):
    """Without opt-in flags, the popgen category is absent (not present as empty)."""
    base = tmp_path / "p"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(base)])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(base)["pedigree"]
    assert "popgen" not in ped
    # None of the categories that *are* present should be empty dicts.
    for cat_name, cat in ped.items():
        assert cat, f"category {cat_name!r} present but empty"


# --------------------------------------------------------------------------
# Direct unit tests on _split_summary / _deep_merge_summary
# --------------------------------------------------------------------------


def test_list_of_dict_split_zips_by_index():
    """A list-of-dict section splits row-by-row; deep-merge zips by index."""
    nested = {
        "strata": {
            "generation_summary": [
                {"gen": 0, "n": 50, "mean_F": 0.0, "n_inbred": 0},
                {"gen": 1, "n": 75, "mean_F": 0.012, "n_inbred": 5},
            ],
        },
    }
    slim, extra = _ps._split_summary(nested)
    # Slim rows keep only the slim_keys ("gen", "n").
    assert slim["strata"]["generation_summary"] == [
        {"gen": 0, "n": 50}, {"gen": 1, "n": 75},
    ]
    # Extra rows carry the residue, aligned by index.
    extra_rows = extra["strata"]["generation_summary"]
    assert extra_rows[0] == {"mean_F": 0.0, "n_inbred": 0}
    assert extra_rows[1] == {"mean_F": 0.012, "n_inbred": 5}
    # Round-trip: deep-merge restores the original.
    merged = _ps._deep_merge_summary(slim, extra)
    assert merged == nested


def test_schema_no_overlap_between_slim_and_extra(tmp_path):
    """No leaf path lives in both slim and extra (totality / no-duplication).

    Walks every leaf of slim's ``pedigree`` subtree and every leaf of
    extra's ``pedigree`` subtree (likewise for ``individual``), collects
    fully-qualified dotted paths (with ``[i]`` for list indices), and
    asserts the intersection is empty.
    """
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(base)
    extra = _load_extra(base)

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
        assert not overlap, (
            f"{top}: {len(overlap)} leaf(s) appear in both slim and extra: "
            f"{sorted(overlap)[:5]}"
        )


def test_known_yaml_drops_absent_from_both_files(tmp_path):
    """``KNOWN_YAML_DROPS`` paths must not appear in slim or extra."""
    base = tmp_path / "p"
    res = _run([
        "summarize", "--in", str(EXAMPLE), "--out", str(base),
        "--inbreeding", "--effective-size",
    ])
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(base)
    extra = _load_extra(base)

    # pairs.by_degree must be gone from both files.
    slim_pairs = slim["pedigree"]["relatedness"]["pairs"]
    extra_pairs = extra["pedigree"].get("relatedness", {}).get("pairs", {})
    assert "by_degree" not in slim_pairs
    assert "by_degree" not in extra_pairs

    # individual.distributions.F.max must be gone from extra (slim never
    # carried it because the slim-key whitelist for F is mean+median only).
    extra_F = extra["individual"].get("distributions", {}).get("F", {})
    assert "max" not in extra_F


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
