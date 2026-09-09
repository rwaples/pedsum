"""End-to-end tests for the ``--effective-size`` CLI flag in pedsum.

Drives the CLI ``_run_summarize`` path on the bundled
``example_pedigree.tsv`` and asserts on the resulting YAML / TSV.
Pins the TSV/YAML split, the flag-combination warning, and the
observed-label shape of the per-depth arrays.

Pedsum 0.7: ``--effective-size`` is on by default; ``--no-effective-size``
opts out. Long-form TSV (``summary.pedigree.tsv``) is opt-in via ``--tsv``.
"""

from __future__ import annotations

from conftest import (
    EXAMPLE,
)
from conftest import (
    load_summary_extra_yaml as _load_extra,
)
from conftest import (
    load_summary_tsv as _load_tsv,
)
from conftest import (
    load_summary_yaml as _load_yaml,
)
from conftest import (
    run_pedsum as _run,
)


def test_no_effective_size_omits_keys(tmp_path):
    """``--no-effective-size`` skips the Ne sections entirely."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--no-effective-size",
            "--tsv",
        ]
    )
    assert res.returncode == 0, res.stderr

    ped = _load_yaml(out_dir)["pedigree"]
    # popgen category is omitted entirely when effective_size is not computed.
    assert "popgen" not in ped
    assert "effective_size_scalars" not in ped

    tsv_rows = _load_tsv(out_dir)
    assert not any(r[0].startswith("effective_size") for r in tsv_rows[1:])


def test_default_run_populates_effective_size(tmp_path):
    """Bare ``summarize`` populates Ne by default (opt-out)."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr

    ped = _load_yaml(out_dir)["pedigree"]
    assert "effective_size" in ped["popgen"]
    assert len(ped["popgen"]["effective_size"]) == 8


def test_effective_size_without_inbreeding_works(tmp_path):
    """Decoupling check: ``--no-inbreeding`` alone keeps Ne on, omits F."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--no-inbreeding",
        ]
    )
    assert res.returncode == 0, res.stderr

    ped = _load_yaml(out_dir)["pedigree"]
    # Without --inbreeding, the inbreeding section is absent.
    assert "inbreeding" not in ped.get("relatedness", {})
    assert "effective_size" in ped["popgen"]
    assert len(ped["popgen"]["effective_size"]) == 8
    # Without --ne-coancestry the Ne_C scalar is null (compact stub) and the
    # stub says why rather than leaving the absence unexplained.
    assert ped["popgen"]["effective_size"]["ne_coancestry"] == {"ne": None, "reason": "not_requested"}


def test_effective_size_with_inbreeding(tmp_path):
    """Bare run populates both sections; memo_size is gone."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr

    ped = _load_yaml(out_dir)["pedigree"]
    assert ped["relatedness"]["inbreeding"] is not None
    # memo_size dropped from the inbreeding summary in the refactor.
    assert "memo_size" not in ped["relatedness"]["inbreeding"]
    assert ped["popgen"]["effective_size"]["ne_coancestry"]["ne"] is None


def test_ne_coancestry_opts_in(tmp_path):
    """``--ne-coancestry`` enables the coancestry-based Ne computation."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--ne-coancestry",
        ]
    )
    assert res.returncode == 0, res.stderr

    es = _load_yaml(out_dir)["pedigree"]["popgen"]["effective_size"]
    ne_c = es["ne_coancestry"]["ne"]
    # On the 200-row example we expect a finite scalar; if upstream changes
    # cause it to come back as None for genuine reasons, the test still
    # validates "not the NaN-array sentinel that --skip-ne-coancestry=True
    # would have produced".
    assert ne_c is None or isinstance(ne_c, float)
    # The per-gen array lives in the extra YAML.
    extra_es = _load_extra(out_dir)["pedigree"]["popgen"]["effective_size"]
    assert all(v is not None for v in extra_es["ne_coancestry"]["mean_theta_per_gen"][1:])


def test_observed_depth_labels_index_the_arrays(tmp_path):
    """Ne records carry the depth labels they observed, and rates are transition-aligned.

    pedigree-graph 0.8 indexes each record by the labels it actually saw
    instead of a dense ``0 .. max`` range, and a rate between consecutive
    depths has one fewer entry than the depths themselves.
    """
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--ne-coancestry"])
    assert res.returncode == 0, res.stderr

    extra_es = _load_extra(out_dir)["pedigree"]["popgen"]["effective_size"]
    for name in ("ne_inbreeding", "ne_coancestry", "ne_caballero_toro"):
        record = extra_es[name]
        depths = record["depths"]
        assert depths == sorted(set(depths))
        assert len(record["transition_from"]) == len(depths) - 1
        assert len(record["transition_to"]) == len(depths) - 1
        assert len(record["ne_per_gen"]) == len(depths) - 1

    # Ne_V is indexed by the *parent* depth of each transition, so its per-
    # transition array is as long as that label array, not one shorter.
    ne_v = extra_es["ne_variance_family_size"]
    assert len(ne_v["ne_per_transition"]) == len(ne_v["parent_depths"])

    # Per-depth (not per-transition) arrays stay aligned with the labels.
    ne_sr = extra_es["ne_sex_ratio"]
    assert len(ne_sr["ne_per_gen"]) == len(ne_sr["depths"])


def test_unavailable_estimator_still_emits_a_scalar_row(tmp_path):
    """All eight scalar rows survive when an estimator reports no value.

    ``ne_coancestry`` is unselected by default and ``ne_long_term_contributions``
    reports no ``ne`` on this pedigree (its contributions have not reached an
    asymptote), so the projection has to read a missing ``ne`` as null rather
    than dropping the row.
    """
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--tsv"])
    assert res.returncode == 0, res.stderr

    ped = _load_yaml(out_dir)["pedigree"]
    assert "reason" in ped["popgen"]["effective_size"]["ne_coancestry"]
    assert ped["popgen"]["effective_size"]["ne_long_term_contributions"]["ne"] is None

    scalar_rows = [r for r in _load_tsv(out_dir)[1:] if r[0] == "effective_size_scalars"]
    assert len(scalar_rows) == 8
    named = {r[1]: r[3] for r in scalar_rows}
    assert named["ne_coancestry"] == ""
    assert named["ne_long_term_contributions"] == ""


def test_tsv_split_holds(tmp_path):
    """With ``--tsv``, only the eight scalar Ne values appear in the long-form TSV."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--tsv",
        ]
    )
    assert res.returncode == 0, res.stderr

    tsv_rows = _load_tsv(out_dir)
    scalar_rows = [r for r in tsv_rows[1:] if r[0] == "effective_size_scalars"]
    deep_rows = [r for r in tsv_rows[1:] if r[0] == "effective_size"]
    assert len(scalar_rows) == 8
    assert len(deep_rows) == 0


def test_warning_for_orphaned_ne_coancestry(tmp_path):
    """``--ne-coancestry`` alongside ``--no-effective-size`` warns on stderr."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--no-effective-size",
            "--ne-coancestry",
        ]
    )
    assert res.returncode == 0
    assert "has no effect under --no-effective-size" in res.stderr


def test_ne_threads_flag_is_gone(tmp_path):
    """``--ne-threads`` was removed; argparse rejects it instead of silently ignoring it."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--ne-threads", "4"])
    assert res.returncode != 0
    assert "--ne-threads" in res.stderr
    assert not (out_dir / "summary.yaml").exists()
