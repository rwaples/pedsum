"""``validate --break-role-conflicts``: a sexed individual used as both mother and father (ADR 0006).

pedigree-graph 0.15 refuses an id in both parent roles. The repair keeps the
links the recorded sex agrees with and clears the rest; without it the
individual BLOCKs (``parent_role_conflict``).
"""

from __future__ import annotations

import polars as pl
from conftest import load_validate_tsv_gz, run_check, run_pedsum

from pedsum.validate import ValidationConfig

REPAIR = ValidationConfig(break_role_conflicts=True)

# fx (recorded F) is the mother of c1 and the father of c2 and c3.
TEXT = (
    "id\tsex\tmother\tfather\n"
    "m0\tM\t-1\t-1\nf0\tF\t-1\t-1\nfx\tF\t-1\t-1\n"
    "c1\tM\tfx\tm0\nc2\tF\tf0\tfx\nc3\tM\tf0\tfx\n"
)


def _validate(tmp_path, text, *extra):
    ped = tmp_path / "p.tsv"
    ped.write_text(text)
    out = tmp_path / "out"
    return run_pedsum(["validate", "--in", str(ped), "--out", str(out), *extra]), out


def _cleared(out) -> list[tuple[str, ...]]:
    return pl.read_csv(out / "validate.cleared_links.tsv", separator="\t", infer_schema=False).rows()


def test_recorded_female_loses_her_father_links(tmp_path):
    """The father links are cleared and listed; every individual stays; summarize runs on the output."""
    r, out = _validate(tmp_path, TEXT, "--break-role-conflicts")
    assert r.returncode == 1, r.stderr
    reason = "father recorded F and also used as mother"
    assert _cleared(out) == [("c2", "father", "fx", reason), ("c3", "father", "fx", reason)]
    written = load_validate_tsv_gz(out)
    assert sorted(written["id"].to_list()) == ["c1", "c2", "c3", "f0", "fx", "m0"]
    assert written.filter(pl.col("id") == "c1")["mother"].item() == "fx"
    assert written.filter(pl.col("id").is_in(["c2", "c3"]))["father"].to_list() == [None, None]
    s = run_pedsum(["summarize", "--in", str(out / "validate.tsv.gz"), "--out", str(tmp_path / "s")])
    assert s.returncode == 0, s.stderr


def test_recorded_male_loses_his_mother_links():
    """A recorded M keeps his father links."""
    rows = [
        {"id": "1", "sex": "M", "mother": "-1", "father": "-1"},
        {"id": "2", "sex": "F", "mother": "-1", "father": "-1"},
        {"id": "3", "sex": "F", "mother": "1", "father": "-1"},
        {"id": "4", "sex": "F", "mother": "2", "father": "1"},
    ]
    result = run_check("role_conflict_breaks", rows, REPAIR)
    assert [f.detail for f in result.findings] == [
        "id=3: cleared mother id=1, mother recorded M and also used as father"
    ]


def test_without_the_flag_it_blocks(tmp_path):
    """Plain validate BLOCKs, naming the rows and the flag."""
    r, out = _validate(tmp_path, TEXT)
    assert r.returncode == 2, r.stderr
    assert "--break-role-conflicts" in r.stderr
    log = pl.read_csv(out / "validate.log", separator="\t", infer_schema=False)
    assert log.rows() == [
        (
            "parent_role_conflict",
            "fx",
            "2",
            "id=fx recorded F, used as mother in row(s) [3] and as father in row(s) [4, 5]",
            None,
        )
    ]


def test_unsexed_is_never_repaired():
    """No recorded sex, no evidence: sex_role_ambiguity owns it and nothing is cleared."""
    rows = [
        {"id": "1", "sex": "", "mother": "-1", "father": "-1"},
        {"id": "2", "sex": "M", "mother": "1", "father": "-1"},
        {"id": "3", "sex": "F", "mother": "-1", "father": "1"},
    ]
    assert run_check("role_conflict_breaks", rows, REPAIR).status == "PASS"
    assert run_check("parent_role_conflict", rows, REPAIR).status == "PASS"
    assert run_check("sex_role_ambiguity", rows, REPAIR).status == "FAIL"


def test_both_parents_of_one_row_is_left_to_parents_distinct():
    """A row whose mother and father are the same id keeps both links; parents_distinct still FAILs."""
    rows = [
        {"id": "1", "sex": "F", "mother": "-1", "father": "-1"},
        {"id": "2", "sex": "M", "mother": "-1", "father": "-1"},
        {"id": "3", "sex": "F", "mother": "1", "father": "1"},
        {"id": "4", "sex": "F", "mother": "1", "father": "2"},
        {"id": "5", "sex": "F", "mother": "-1", "father": "1"},
    ]
    result = run_check("role_conflict_breaks", rows, REPAIR)
    assert [f.detail for f in result.findings] == [
        "id=5: cleared father id=1, father recorded F and also used as mother"
    ]
    assert run_check("parents_distinct", rows, REPAIR).status == "FAIL"
    # Used as a mother only by its own selfed row: not in both roles here.
    selfed = [rows[0], rows[1], rows[2], rows[4]]
    assert run_check("role_conflict_breaks", selfed, REPAIR).status == "PASS"
    assert run_check("parent_role_conflict", selfed).status == "PASS"


def test_a_repair_after_a_dropped_duplicate_is_in_the_manifest(tmp_path):
    """A duplicate id gates the role checks in round 1; the round-2 repair still reaches the manifest."""
    text = TEXT + "d\tM\t-1\t-1\nd\tF\t-1\t-1\n"
    r, out = _validate(tmp_path, text, "--break-role-conflicts", "--drop-offending")
    assert r.returncode == 1, r.stderr
    assert [link[:3] for link in _cleared(out)] == [("c2", "father", "fx"), ("c3", "father", "fx")]
    assert load_validate_tsv_gz(out).filter(pl.col("id") == "c2")["father"].item() is None


def test_drop_offending_with_allow_missing_sex_keeps_an_unsexed_parent_in_both_roles(tmp_path):
    """--allow-missing-sex tolerates the ambiguity, so --drop-offending has nothing of it to drop."""
    text = "id\tsex\tmother\tfather\nm0\tM\t-1\t-1\nf0\tF\t-1\t-1\nu\t\t-1\t-1\nc1\tM\tu\tm0\nc2\tF\tf0\tu\n"
    r, out = _validate(tmp_path, text, "--allow-missing-sex", "--drop-offending")
    assert r.returncode == 0, r.stderr
    assert "u" in load_validate_tsv_gz(out)["id"].to_list()


def test_no_sex_check_founder_blocks_when_read_back(tmp_path):
    """--no-sex-check writes a missing parent in both roles as an F founder, which then BLOCKs as a role conflict."""
    text = "id\tsex\tmother\tfather\nm0\tM\t-1\t-1\nf0\tF\t-1\t-1\nc1\tM\tx\tm0\nc2\tF\tf0\tx\n"
    r, out = _validate(tmp_path, text, "--no-sex-check")
    assert r.returncode == 1, r.stderr
    written = load_validate_tsv_gz(out)
    assert written.filter(pl.col("id") == "x")["sex"].item() == "2"  # PLINK female
    again = run_pedsum(["validate", "--in", str(out / "validate.tsv.gz"), "--out", str(tmp_path / "again")])
    assert again.returncode == 2, again.stderr
    assert "--break-role-conflicts" in again.stderr
