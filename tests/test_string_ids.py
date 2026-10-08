"""String IDs end to end: every output writes the IDs the input wrote (issue #16).

A file whose ID tokens are not all plain integers is read in string mode: two
tokens are the same ID only if they are the same string. See ``pedsum.ids``.
"""

from __future__ import annotations

import polars as pl
from conftest import load_validate_tsv_gz, read_tsv_gz, run_pedsum

from pedsum.ids import PHANTOM_PREFIX


def _write(path, text: str):
    path.write_text(text)
    return path


def _links_resolve(df: pl.DataFrame) -> bool:
    """Every non-missing parent token names a row, comparing IDs as strings."""
    ids = set(df["id"].to_list())
    parents = [p for col in ("mother", "father") for p in df[col].to_list() if p not in (None, "-1")]
    return all(p in ids for p in parents)


def test_issue_example_keeps_tokens_distinct(tmp_path):
    """``001``, ``1``, ``1.0`` and ``2.9`` are four IDs; the written file's links resolve as strings."""
    ped = _write(
        tmp_path / "p.tsv", "id\tmother\tfather\tsex\n001\t-1\t-1\t2\n2\t-1\t-1\t1\n3\t1\t2\t2\n4\t1.0\t2.9\t1\n"
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--allow-missing-sex"])
    assert r.returncode == 1, r.stderr  # parents 1, 1.0 and 2.9 have no row
    assert "IDs read as strings" in r.stderr
    log = pl.read_csv(out / "validate.log", separator="\t", infer_schema=False)
    missing = log.filter(pl.col("check").str.starts_with("parent_refs_present"))
    assert sorted(missing["id"].to_list()) == ["1", "1.0", "2.9"]
    fixed = load_validate_tsv_gz(out)
    assert sorted(fixed["id"].to_list()) == ["001", "1", "1.0", "2", "2.9", "3", "4"]
    assert _links_resolve(fixed)


def _alnum_pedigree(path):
    return _write(
        path,
        "id\tsex\tmother\tfather\n"
        "F01\tF\tNA\tNA\n"
        "M01\tM\tNA\tNA\n"
        "c-1\tF\tF01\tM01\n"
        "c-2\tM\tF01\tM01\n"
        "g.1\tM\tc-1\tM01\n",
    )


def test_alphanumeric_ids_validate_clean_and_round_trip(tmp_path):
    """Alphanumeric IDs pass validate, and validate.tsv.gz writes them unchanged."""
    ped = _alnum_pedigree(tmp_path / "p.tsv")
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out)])
    assert r.returncode == 0, r.stderr
    fixed = load_validate_tsv_gz(out)
    assert fixed["id"].to_list() == ["F01", "M01", "c-1", "c-2", "g.1"]
    assert _links_resolve(fixed)
    r2 = run_pedsum(["validate", "--in", str(out / "validate.tsv.gz"), "--out", str(tmp_path / "out2")])
    assert r2.returncode == 0, r2.stderr


def _findings(tmp_path, text: str) -> pl.DataFrame:
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(_write(tmp_path / "p.tsv", text)), "--out", str(out)])
    assert r.returncode == 2, r.stderr  # both findings block
    return pl.read_csv(out / "validate.log", separator="\t", infer_schema=False)


def test_findings_name_string_ids(tmp_path):
    """Findings name string IDs in validate.log's id column and detail text."""
    log = _findings(tmp_path, "id\tsex\tmother\tfather\nA\tF\t-1\t-1\nB\tF\tB\t-1\n")
    assert log.filter(pl.col("check") == "self_loops")["id"].to_list() == ["B"]
    assert "row 1: id=B listed as own mother" in log["detail"].to_list()


def test_duplicate_string_ids(tmp_path):
    """The same string twice is a duplicate; ``A`` and ``a`` are two IDs."""
    log = _findings(tmp_path, "id\tsex\tmother\tfather\nA\tF\t-1\t-1\na\tF\t-1\t-1\nA\tM\t-1\t-1\n")
    assert log.filter(pl.col("check") == "duplicate_ids")["id"].to_list() == ["A"]


def test_summarize_and_epimight_write_string_ids(tmp_path):
    """annotated.tsv.gz, pipeline_input.tsv and relative_pairs.tsv carry the input's IDs."""
    ped = _alnum_pedigree(tmp_path / "p.tsv")
    r = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "s"), "--no-inbreeding"])
    assert r.returncode == 0, r.stderr
    ann = read_tsv_gz(tmp_path / "s" / "annotated.tsv.gz", as_str=True)
    assert ann["id"].to_list() == ["F01", "M01", "c-1", "c-2", "g.1"]
    by_id = {row["id"]: row for row in ann.iter_rows(named=True)}
    assert (by_id["g.1"]["mother"], by_id["g.1"]["father"]) == ("c-1", "M01")
    assert by_id["F01"]["mother"] == "-1"

    epi = tmp_path / "e"
    r = run_pedsum(["epimight-input", "--in", str(ped), "--out", str(epi), "--pairs", "--rels", "FS,PO"])
    assert r.returncode == 0, r.stderr
    skeleton = pl.read_csv(epi / "pipeline_input.tsv", separator="\t", infer_schema=False)
    assert set(skeleton["person_id"]) == {"F01", "M01", "c-1", "c-2", "g.1"}
    pairs = pl.read_csv(epi / "relative_pairs.tsv", separator="\t", infer_schema=False)
    fs = pairs.filter(pl.col("relationship_kind") == "FS")
    assert list(zip(fs["id1"], fs["id2"], strict=True)) == [("c-1", "c-2")]
    assert ("g.1", "c-1") in set(zip(pairs["id1"], pairs["id2"], strict=True))


def test_effective_size_reference_column_matches_string_ids(tmp_path):
    """--reference-col finds its rows by string ID."""
    ped = _write(
        tmp_path / "p.tsv",
        "id\tsex\tmother\tfather\tref\n"
        "F01\tF\tNA\tNA\t0\nM01\tM\tNA\tNA\t0\nc-1\tF\tF01\tM01\t1\nc-2\tM\tF01\tM01\t1\n",
    )
    out = tmp_path / "es"
    args = ["--estimators", "ne_individual_delta_f", "--reference-col", "ref"]
    r = run_pedsum(["effective-size", "--in", str(ped), "--out", str(out), *args])
    assert r.returncode == 0, r.stderr
    assert "reference" in (out / "effective_size.yaml").read_text()


def test_drop_offending_manifest_names_string_ids(tmp_path):
    """The removal manifest lists dropped IDs as the input wrote them."""
    ped = _write(
        tmp_path / "p.tsv",
        "id\tsex\tmother\tfather\nm1\tF\t-1\t-1\nf1\tM\t-1\t-1\nboth\tF\t-1\t-1\nk1\tF\tboth\tf1\nk2\tM\tm1\tboth\n",
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending"])
    assert r.returncode == 1, r.stderr
    manifest = pl.read_csv(out / "validate.dropped.tsv", separator="\t", infer_schema=False)
    assert manifest["id"].to_list() == ["both"]
    assert "both" not in load_validate_tsv_gz(out)["id"].to_list()


def test_fill_half_founders_names_phantoms_with_the_prefix(tmp_path):
    """String-mode phantom parents get prefixed IDs that the written file links to."""
    ped = _write(tmp_path / "p.tsv", "id\tsex\tmother\tfather\nm1\tF\t-1\t-1\nk1\tF\tm1\t-1\nk2\tM\t-1\t-1\n")
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--fill-half-founders"])
    assert r.returncode == 0, r.stderr
    fixed = load_validate_tsv_gz(out)
    by_id = {row["id"]: row for row in fixed.iter_rows(named=True)}
    assert by_id["k1"]["father"] == f"{PHANTOM_PREFIX}1"
    assert by_id[f"{PHANTOM_PREFIX}1"]["sex"] == "1"
    assert _links_resolve(fixed)


def test_fill_half_founders_refuses_ids_with_the_phantom_prefix(tmp_path):
    """An input ID starting with the phantom prefix could collide, so the fill refuses."""
    ped = _write(
        tmp_path / "p.tsv",
        f"id\tsex\tmother\tfather\n{PHANTOM_PREFIX}1\tF\t-1\t-1\nk1\tF\t{PHANTOM_PREFIX}1\t-1\n",
    )
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "out"), "--fill-half-founders"])
    assert r.returncode == 2
    assert "already has IDs with that prefix" in r.stderr
    assert not (tmp_path / "out" / "validate.tsv.gz").exists()


def test_missing_token_as_an_id_is_rejected(tmp_path):
    """An id that is a missing-parent token (``.``) could never be named as a parent, so it is a missing id."""
    ped = _write(tmp_path / "p.tsv", "id\tsex\tmother\tfather\n.\tF\t-1\t-1\nb\tM\t-1\t-1\nc\tM\t.\tb\n")
    out = tmp_path / "out"
    run_pedsum(["validate", "--in", str(ped), "--out", str(out)])
    log = pl.read_csv(out / "validate.log", separator="\t", infer_schema=False)
    assert log["check"].to_list() == ["id_dtype"]
    assert "1 missing value(s)" in log["detail"][0]


def test_drop_offending_keeps_string_mode_when_the_last_string_id_is_dropped(tmp_path):
    """Dropping the only non-integer ID does not switch the reduction to integer mode.

    Regression: the phantom then took a code equal to the real id 3, and
    validate.tsv.gz had two rows with id 3.
    """
    ped = _write(
        tmp_path / "p.tsv",
        "id\tsex\tmother\tfather\n2\tF\t-1\t-1\nx\tM\t-1\t-1\nx\tM\t-1\t-1\n3\tM\t2\t-1\n",
    )
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--drop-offending", "--fill-half-founders"])
    assert r.returncode == 1, r.stderr  # x was dropped; self-verify passed
    fixed = load_validate_tsv_gz(out)
    assert sorted(fixed["id"].to_list()) == ["2", "3", f"{PHANTOM_PREFIX}1"]
    assert fixed.filter(pl.col("id") == "3")["father"][0] == f"{PHANTOM_PREFIX}1"
    assert r.stderr.count("(e.g. ['x'])") == 1  # the rounds reuse the input's labels
