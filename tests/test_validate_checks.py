"""Behavior tests for the validate subcommand: hard-blocks, summary layout."""

from __future__ import annotations

import gzip

import polars as pl
import pytest
from conftest import run_pedsum
from conftest import write_ped as _write_ped


def test_validate_unknown_sex_is_hard_block(tmp_path):
    """Orphan unsexed row without --allow-missing-sex hard-blocks validate."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "", "mother": 2, "father": 1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 2, r.stderr
    assert "BLOCKED" in r.stderr
    assert "unresolved sex" in r.stderr.lower() or "unknown" in r.stderr.lower()
    # .validate.tsv.gz must NOT be written when blocked
    assert not (tmp_path / "out" / "validate.tsv.gz").exists()


def test_validate_unknown_sex_skipped_with_flag(tmp_path):
    """--allow-missing-sex turns the unknown_sex check into a SKIP."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "", "mother": 2, "father": 1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(base),
            "--allow-missing-sex",
        ]
    )
    assert r.returncode == 0, r.stderr
    # Summary shows SKIP for unknown_sex with the tolerated count.
    assert "SKIP" in r.stderr
    assert "tolerated" in r.stderr
    # Output file IS written.
    assert (tmp_path / "out" / "validate.tsv.gz").exists()
    # validate.log does NOT contain rows for tolerated unknown_sex findings
    # (matches existing precedent: _check_* is skipped entirely when flagged).
    log = (tmp_path / "out" / "validate.log").read_text()
    assert "unknown_sex" not in log


def _ambig_pedigree(path):
    """Pedigree where id=7 is unsexed and used as both mother and father."""
    return _write_ped(
        path,
        [
            {"id": 7, "sex": "", "mother": -1, "father": -1},
            {"id": 8, "sex": "M", "mother": -1, "father": -1},
            {"id": 9, "sex": "F", "mother": -1, "father": -1},
            {"id": 10, "sex": "F", "mother": 7, "father": 8},
            {"id": 11, "sex": "M", "mother": 9, "father": 7},
        ],
    )


def test_validate_sex_role_ambiguity_is_hard_block_without_flag(tmp_path):
    """sex_role_ambiguity hard-blocks validate when --allow-missing-sex is absent."""
    ped = _ambig_pedigree(tmp_path / "p.tsv")
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 2, r.stderr
    assert "BLOCKED" in r.stderr
    assert not (tmp_path / "out" / "validate.tsv.gz").exists()


def test_validate_sex_role_ambiguity_passes_with_flag(tmp_path):
    """--allow-missing-sex lets ambiguity through; fixed TSV writes sex=0 (unknown) for the row."""
    import gzip

    ped = _ambig_pedigree(tmp_path / "p.tsv")
    base = tmp_path / "out"
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(base),
            "--allow-missing-sex",
        ]
    )
    assert r.returncode == 0, r.stderr
    # SKIP message in the grouped summary; no BLOCKED.
    assert "BLOCKED" not in r.stderr
    assert "tolerated" in r.stderr
    fixed = tmp_path / "out" / "validate.tsv.gz"
    assert fixed.exists()
    with gzip.open(fixed, "rb") as fh:
        df = pl.read_csv(fh.read(), separator="\t", infer_schema=False)
    row = df.filter(pl.col("id").cast(pl.Int64) == 7).row(0, named=True)
    assert row["sex"] == "0"
    # validate.log should NOT contain a row for the tolerated ambiguous id.
    log = (tmp_path / "out" / "validate.log").read_text()
    assert "sex_role_ambiguity" not in log


def test_sex_role_consistency_log_pins_offending_rows(tmp_path):
    """sex_role_consistency names the lines, not just the id.

    id=3 (row 2, female) is used as a father in rows 4 and 5. The finding now
    carries the id's own row in the ``row`` column and the referencing rows in
    the detail, so the log is actionable without grepping the input.
    """
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},
            {"id": 3, "sex": "F", "mother": -1, "father": -1},  # row 2
            {"id": 4, "sex": "F", "mother": 3, "father": 2},  # 3 as mother (consistent)
            {"id": 5, "sex": "M", "mother": 1, "father": 3},  # row 4: 3 as father
            {"id": 6, "sex": "M", "mother": 1, "father": 3},  # row 5: 3 as father
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 1, r.stderr  # FAIL but not a hard block
    log = pl.read_csv(base / "validate.log", separator="\t")
    finding = log.filter(pl.col("check") == "sex_role_consistency")
    assert len(finding) == 1
    row = finding.row(0, named=True)
    assert int(row["id"]) == 3
    assert int(row["row"]) == 2  # id=3's own row (where its sex lives)
    assert "row(s) [4, 5]" in row["detail"]


def test_parent_refs_sex_conflict_log_pins_rows(tmp_path):
    """parent_refs_sex_conflict names the mother and father referencing rows.

    Missing parent id=99 is referenced as mother in rows 2 and 4 and as father
    in row 3; the detail now names both sets of lines.
    """
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},
            {"id": 4, "sex": "F", "mother": 99, "father": 2},  # row 2: 99 as mother
            {"id": 5, "sex": "M", "mother": 1, "father": 99},  # row 3: 99 as father
            {"id": 7, "sex": "F", "mother": 99, "father": 2},  # row 4: 99 as mother
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 2, r.stderr  # hard block
    log = pl.read_csv(base / "validate.log", separator="\t")
    finding = log.filter(pl.col("check") == "parent_refs_sex_conflict")
    assert len(finding) == 1
    detail = finding.row(0, named=True)["detail"]
    assert "mother (row(s) [2, 4])" in detail
    assert "father (row(s) [3])" in detail


def test_orphan_unsexed_writes_unknown_in_fixed_tsv(tmp_path):
    """Orphan-only pedigree (n_imputed==0) + --allow-missing-sex: orphan sex normalised to 0 (unknown)."""
    import gzip

    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "", "mother": 2, "father": 1},  # orphan, no role
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(
        [
            "validate",
            "--in",
            str(ped),
            "--out",
            str(base),
            "--allow-missing-sex",
        ]
    )
    assert r.returncode == 0, r.stderr
    fixed = tmp_path / "out" / "validate.tsv.gz"
    assert fixed.exists()
    with gzip.open(fixed, "rb") as fh:
        df = pl.read_csv(fh.read(), separator="\t", infer_schema=False)
    row = df.filter(pl.col("id").cast(pl.Int64) == 3).row(0, named=True)
    assert row["sex"] == "0"


def test_validate_summary_uses_grouped_layout(tmp_path):
    """The validate stderr summary prints the four section headers in order."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 3, "sex": "M", "mother": 2, "father": 1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 0, r.stderr
    expected = [
        "Columns & parsing",
        "IDs",
        "Parent references",
        "Graph structure",
    ]
    last = -1
    for hdr in expected:
        idx = r.stderr.find(hdr)
        assert idx > last, f"section header {hdr!r} missing or out of order"
        last = idx
    # Dot-padded labels: lines like "required columns present .... PASS"
    assert ".. PASS" in r.stderr


def test_validate_writes_imputed_sex_in_fixed_output(tmp_path):
    """The .validate.tsv.gz contains the imputed sex value (F/M) not the blank."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "", "mother": -1, "father": -1},  # imputed F (used as mother)
            {"id": 3, "sex": "F", "mother": 2, "father": 1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 0, r.stderr
    with gzip.open(tmp_path / "out" / "validate.tsv.gz", "rb") as fh:
        fixed = pl.read_csv(fh.read(), separator="\t", infer_schema=False)
    row2 = fixed.filter(pl.col("id").cast(pl.Int64) == 2).row(0, named=True)
    assert row2["sex"] == "2"


def test_parent_refs_present_log_counts_references(tmp_path):
    """validate.log's n_refs counts the rows naming each missing parent; other checks leave it empty.

    Father 900 is named in rows 2 and 3, mother 800 in row 4; id=1 (sex F) is
    also used as a father, so a sex_role_consistency finding shares the log.
    """
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "F", "mother": -1, "father": -1},
            {"id": 2, "sex": "M", "mother": -1, "father": -1},
            {"id": 5, "sex": "F", "mother": 1, "father": 900},
            {"id": 6, "sex": "M", "mother": 1, "father": 900},
            {"id": 7, "sex": "M", "mother": 800, "father": 1},
        ],
    )
    base = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(base)])
    assert r.returncode == 1, r.stderr
    log = pl.read_csv(base / "validate.log", separator="\t")
    assert log.schema["n_refs"] == pl.Int64
    refs = log.filter(pl.col("check").str.starts_with("parent_refs_present"))
    assert sorted(refs.select("check", "id", "n_refs").rows()) == [
        ("parent_refs_present_father", 900, 2),
        ("parent_refs_present_mother", 800, 1),
    ]
    others = log.filter(~pl.col("check").str.starts_with("parent_refs_present"))
    assert "sex_role_consistency" in others["check"].to_list()
    assert others["n_refs"].null_count() == len(others)


@pytest.mark.parametrize(
    ("text", "check"),
    [
        # No father column.
        ("id\tmother\tsex\n1\t-1\tM\n2\t-1\tF\n3\t2\tF\n", "required_columns"),
        # A row without an id.
        ("id\tmother\tfather\tsex\n\t-1\t-1\tM\n2\t-1\t-1\tF\n", "id_dtype"),
        # An unrecognized sex token.
        ("id\tmother\tfather\tsex\n1\t-1\t-1\tM\n2\t-1\t-1\tQ\n", "sex_tokens"),
        # A negative id.
        ("id\tmother\tfather\tsex\n1\t-1\t-1\tM\n-5\t-1\t-1\tF\n", "negative_ids"),
    ],
)
def test_validate_column_level_failure_blocks(tmp_path, text, check):
    """A column- or parse-level FAIL BLOCKs plain validate, as under --drop-offending (pedsum#21)."""
    ped = tmp_path / "p.tsv"
    ped.write_text(text)
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out)])
    assert r.returncode == 2, r.stderr
    assert "BLOCKED" in r.stderr
    assert f"  - {check}\n" in r.stderr
    assert not (out / "validate.tsv.gz").exists()
    assert (out / "validate.log").exists()


# A cycle (1 and 2 are each other's mother) plus a father ID (99) missing from the file.
_CYCLE_WITH_MISSING_PARENT = "id\tmother\tfather\tsex\n1\t2\t3\tF\n2\t1\t3\tF\n3\t-1\t-1\tM\n4\t-1\t99\tF\n"


def test_cycle_blocks_even_with_a_missing_parent(tmp_path):
    """The cycle Check runs when a parent ID is missing, so the cycle BLOCKs plain validate (pedsum#22)."""
    ped = tmp_path / "p.tsv"
    ped.write_text(_CYCLE_WITH_MISSING_PARENT)
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out)])
    assert r.returncode == 2, r.stderr
    assert "  - cycle detected\n" in r.stderr
    assert not (out / "validate.tsv.gz").exists()


def test_drop_offending_drops_a_cycle_with_a_missing_parent(tmp_path):
    """--drop-offending removes the cycle members and keeps the founder for the missing parent (pedsum#22)."""
    ped = tmp_path / "p.tsv"
    ped.write_text(_CYCLE_WITH_MISSING_PARENT)
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    assert r.returncode == 1, r.stderr
    with gzip.open(out / "validate.tsv.gz", "rt") as fh:
        fixed = pl.read_csv(fh.read().encode(), separator="\t", infer_schema=False)
    assert sorted(fixed["id"].to_list()) == ["3", "4", "99"]
    dropped = pl.read_csv(out / "validate.dropped.tsv", separator="\t", infer_schema=False)
    assert sorted(set(dropped["id"].to_list())) == ["1", "2"]


def test_self_loop_blocks_plain_validate_as_a_cycle(tmp_path):
    """A self-loop is a one-person cycle: acyclic FAILs with self_loops, so plain validate BLOCKs."""
    ped = tmp_path / "p.tsv"
    ped.write_text("id\tmother\tfather\tsex\n1\t-1\t-1\tM\n2\t-1\t-1\tF\n3\t3\t1\tF\n")
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "out")])
    assert r.returncode == 2, r.stderr
    assert "  - cycle detected\n" in r.stderr


def test_blocked_run_removes_a_previous_runs_outputs(tmp_path):
    """A BLOCKED re-run into the same --out leaves no validate.tsv.gz / validate.dropped.tsv from an earlier input."""
    out = tmp_path / "out"
    good = tmp_path / "good.tsv"
    good.write_text("id\tmother\tfather\tsex\n1\t-1\t-1\tM\n2\t-1\t-1\tF\n3\t2\t1\tF\n")
    r = run_pedsum(["validate", "--in", str(good), "--out", str(out), "--drop-offending"])
    assert r.returncode == 0, r.stderr
    assert (out / "validate.tsv.gz").exists()
    assert (out / "validate.dropped.tsv").exists()
    bad = tmp_path / "bad.tsv"
    bad.write_text("id\tmother\tsex\n1\t-1\tM\n")
    r = run_pedsum(["validate", "--in", str(bad), "--out", str(out)])
    assert r.returncode == 2, r.stderr
    assert not (out / "validate.tsv.gz").exists()
    assert not (out / "validate.dropped.tsv").exists()
    assert (out / "validate.log").exists()
