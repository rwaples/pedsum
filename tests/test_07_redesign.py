"""Tests for the pedsum 0.7 CLI redesign.

Pins the breaking changes:

- Deleted flags exit rc=2 with a clear "unrecognized arguments" message.
- argparse abbreviation is disabled, so partial-matches of deleted flags
  also fail (cannot silently resurrect a removed long-option).
- ``--inbreeding`` and ``--effective-size`` are opt-out (default on);
  ``--no-inbreeding`` / ``--no-effective-size`` skip them.
- ``--out DIR`` is a directory; default footprint is 3 files inside.
- ``--tsv`` opts into the two long-form TSV outputs.
"""

from __future__ import annotations

import pytest
from conftest import EXAMPLE
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run

# --- Deleted flags exit rc=2 ----------------------------------------------


@pytest.mark.parametrize(
    ("flag", "args_after"),
    [
        ("--engine", ["matrix"]),
        ("--bfs-threshold", ["1000000"]),
        ("--zero-as-missing", []),
        ("--single-file", []),
        ("--burden", []),
        ("--allow-unknown-sex", []),  # renamed to --allow-missing-sex in 0.8
    ],
)
def test_deleted_flag_exits_rc2(tmp_path, flag, args_after):
    """Each deleted long-option exits rc=2 with an argparse error."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            flag,
            *args_after,
        ]
    )
    assert res.returncode == 2, res.stderr
    assert "unrecognized arguments" in res.stderr
    assert flag in res.stderr


def test_abbreviated_deleted_flag_does_not_resurrect(tmp_path):
    """``--bfs`` should not silently match a surviving flag via abbrev prefix."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--bfs",  # abbreviation of the now-deleted --bfs-threshold
        ]
    )
    assert res.returncode == 2, res.stderr
    assert "unrecognized arguments" in res.stderr


# --- Opt-out defaults ------------------------------------------------------


def test_bare_summarize_emits_inbreeding(tmp_path):
    """Bare ``summarize`` populates the inbreeding section by default."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(out_dir)["pedigree"]
    assert ped["relatedness"]["inbreeding"] is not None


def test_bare_summarize_emits_effective_size(tmp_path):
    """Bare ``summarize`` populates the effective-size section by default."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    ped = _load_yaml(out_dir)["pedigree"]
    assert "effective_size" in ped["popgen"]


def test_no_inbreeding_omits_section(tmp_path):
    """``--no-inbreeding`` skips F and the inbreeding section."""
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
    assert "inbreeding" not in ped.get("relatedness", {})


# --- Output directory + --tsv opt-in --------------------------------------


def test_default_footprint_is_three_files(tmp_path):
    """Bare ``summarize`` writes summary.yaml + summary.extra.yaml + annotated.tsv.gz."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == [
        "annotated.tsv.gz",
        "summary.extra.yaml",
        "summary.yaml",
    ]


def test_tsv_flag_adds_long_form_tsvs(tmp_path):
    """``--tsv`` adds summary.pedigree.tsv and summary.individual.tsv."""
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
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == [
        "annotated.tsv.gz",
        "summary.extra.yaml",
        "summary.individual.tsv",
        "summary.pedigree.tsv",
        "summary.yaml",
    ]


def test_out_path_as_existing_file_errors(tmp_path):
    """Passing ``--out`` that already exists as a regular file exits rc=1."""
    existing = tmp_path / "blocker"
    existing.write_text("not a directory\n")
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(existing),
        ]
    )
    assert res.returncode == 1, res.stderr
    assert "not a directory" in res.stderr


# --- --per-individual-pairs (renamed from --burden) -----------------------


def test_per_individual_pairs_produces_relationship_burden_section(tmp_path):
    """``--per-individual-pairs`` populates the per-individual burden YAML section."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--per-individual-pairs",
        ]
    )
    assert res.returncode == 0, res.stderr
    rs = _load_yaml(out_dir)["pedigree"]["relatedness"]["relationship_summary"]
    assert rs["computed"] is True
    assert "relatives_by_degree" in rs


# --- 0.8: --allow-missing-sex (renamed from --allow-unknown-sex, broadened) ---


def _mixed_pedigree(path):
    """A pedigree with both a role-ambiguous row (id=7) AND an orphan (id=8)."""
    import csv

    rows = [
        {"id": 1, "sex": "M", "mother": -1, "father": -1},
        {"id": 2, "sex": "F", "mother": -1, "father": -1},
        {"id": 7, "sex": "", "mother": -1, "father": -1},  # ambiguous below
        {"id": 8, "sex": "", "mother": -1, "father": -1},  # orphan
        {"id": 3, "sex": "M", "mother": 7, "father": 1},
        {"id": 4, "sex": "F", "mother": 2, "father": 7},
    ]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return path


def test_summarize_missing_sex_with_flag_and_opt_outs(tmp_path):
    """``--allow-missing-sex --no-inbreeding --no-effective-size`` lets summarize finish on a mixed pedigree."""
    import gzip

    import pandas as pd

    ped = _mixed_pedigree(tmp_path / "ped.tsv")
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--allow-missing-sex",
            "--no-inbreeding",
            "--no-effective-size",
        ]
    )
    assert res.returncode == 0, res.stderr
    # Annotated TSV should contain both originally-unsexed rows with sex=-1.
    with gzip.open(out_dir / "annotated.tsv.gz", "rt") as fh:
        ann = pd.read_csv(fh, sep="\t")
    for orig_id in (7, 8):
        row = ann.loc[ann["id"] == orig_id].iloc[0]
        assert int(row["sex"]) == -1, f"id={orig_id} sex={row['sex']}"


def test_summarize_missing_sex_refused_with_inbreeding(tmp_path):
    """Default --inbreeding alongside --allow-missing-sex fires the resolved-sex refusal.

    Pins that the refusal message names the NEW flag (not the deleted
    --allow-unknown-sex) — guards against stale error-message text.
    """
    ped = _mixed_pedigree(tmp_path / "ped.tsv")
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--allow-missing-sex",
        ]
    )
    assert res.returncode == 1, res.stderr
    assert "resolved sex" in res.stderr or "sex-stratified" in res.stderr
    assert "--allow-missing-sex" in res.stderr
    assert "--allow-unknown-sex" not in res.stderr
