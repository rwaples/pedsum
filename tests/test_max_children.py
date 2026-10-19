"""``parent_child_count``: a parent with an implausible number of children, such as a placeholder code (#27)."""

from __future__ import annotations

import polars as pl
from conftest import load_validate_tsv_gz, run_check, run_pedsum, write_ped

from pedsum.validate import ValidationConfig

LIMITS = ValidationConfig(max_children_mother=3, max_children_father=4)
# The limits the help suggests for a human register.
HUMAN = ["--max-children-mother", "20", "--max-children-father", "50"]


def _kids(n: int, mother: str, father: str, start: int = 100) -> list[dict]:
    return [{"id": str(start + k), "sex": "F", "mother": mother, "father": father} for k in range(n)]


def _founders(*ids_sexes: tuple[str, str]) -> list[dict]:
    return [{"id": i, "sex": s, "mother": "", "father": ""} for i, s in ids_sexes]


def _details(rows: list[dict], config: ValidationConfig = LIMITS) -> list[str]:
    return [f.detail for f in run_check("parent_child_count", rows, config).findings]


def _validate(tmp_path, rows: list[dict], *flags: str):
    tmp_path.mkdir(exist_ok=True)
    ped = write_ped(tmp_path / "p.tsv", rows)
    out = tmp_path / "out"
    return run_pedsum(["validate", "--in", str(ped), "--out", str(out), *flags]), out


def test_placeholder_mother_without_a_row_is_flagged():
    """A mother code with no row and too many children is a Finding that names the token to pass."""
    rows = [*_founders(("1", "M")), *_kids(4, "9999", "1")]
    assert _details(rows, ValidationConfig(max_children_mother=3)) == [
        "id=9999 is the mother of 4 children (more than --max-children-mother 3); it has no row, so validate "
        "adds it as a founder. If it stands for an unknown mother, pass --missing-parent-token 9999"
    ]


def test_father_with_a_row_is_told_to_delete_it_first():
    """A token cannot also be an id with a row, so the remedy for a father in the file starts with his row."""
    rows = [*_founders(("1", "M")), *_kids(5, "", "1")]
    findings = run_check("parent_child_count", rows, LIMITS).findings
    assert [(f.id is not None, f.row) for f in findings] == [(True, 0)]
    assert findings[0].detail == (
        "id=1 is the father of 5 children (more than --max-children-father 4); "
        "if it stands for an unknown father, delete its row and pass --missing-parent-token 1"
    )


def test_the_remedy_for_a_placeholder_with_a_row_passes(tmp_path):
    """Deleting the placeholder's row and passing it as a token gives a pedigree that passes."""
    rows = [*_founders(("1", "M"), ("9999", "M")), *_kids(60, "", "9999")]
    r, _ = _validate(tmp_path / "a", rows, *HUMAN)
    assert r.returncode == 1, r.stderr
    r, out = _validate(
        tmp_path / "b", [row for row in rows if row["id"] != "9999"], *HUMAN, "--missing-parent-token", "9999"
    )
    assert r.returncode == 0, r.stderr
    assert load_validate_tsv_gz(out)["father"].is_null().all()


def test_at_the_limit_passes():
    """The limit is the largest count allowed."""
    rows = [*_founders(("1", "M"), ("2", "F")), *_kids(3, "2", "1")]
    assert run_check("parent_child_count", rows, LIMITS).status == "PASS"


def test_zero_turns_a_role_off():
    """0 turns the role's limit off; the other role still counts."""
    rows = [*_founders(("1", "M"), ("2", "F")), *_kids(5, "2", "1")]
    off = ValidationConfig(max_children_mother=0, max_children_father=4)
    assert [d.split(" ", 1)[0] for d in _details(rows, off)] == ["id=1"]
    assert run_check("parent_child_count", rows, ValidationConfig()).status == "SKIP"


def test_off_by_default(tmp_path):
    """Without the flags validate does not count children: a sire of 60 passes."""
    r, out = _validate(tmp_path, [*_founders(("1", "M"), ("2", "F")), *_kids(60, "2", "1")])
    assert r.returncode == 0, r.stderr
    assert "parent_child_count" not in (out / "validate.log").read_text()


def test_real_sibship_of_fifteen_passes_the_human_limits(tmp_path):
    """Fifteen children of one couple are a large family, not a code."""
    r, out = _validate(tmp_path, [*_founders(("1", "M"), ("2", "F")), *_kids(15, "2", "1")], *HUMAN)
    assert r.returncode == 0, r.stderr
    assert "parent_child_count" not in (out / "validate.log").read_text()


def test_validate_flags_a_placeholder_and_writes_it_kept(tmp_path):
    """Plain validate keeps the pedigree as it is and exits 1 with the Finding."""
    r, out = _validate(tmp_path, [*_founders(("1", "M")), *_kids(21, "9999", "1")], *HUMAN)
    assert r.returncode == 1, r.stderr
    log = pl.read_csv(out / "validate.log", separator="\t", infer_schema=False)
    assert "parent_child_count" in log["check"].to_list()
    assert (load_validate_tsv_gz(out)["mother"] == "9999").sum() == 21


def test_drop_offending_drops_the_placeholder_and_keeps_its_children(tmp_path):
    """Dropping the flagged parent clears every child's link to it; the children stay."""
    rows = [*_founders(("1", "M")), *_kids(21, "9999", "1")]
    r, out = _validate(tmp_path, rows, *HUMAN, "--drop-offending")
    assert r.returncode == 1, r.stderr
    written = load_validate_tsv_gz(out)
    assert sorted(written["id"].to_list()) == sorted(["1", *(str(100 + k) for k in range(21))])
    assert not (written["mother"] == "9999").any()
    assert (written["father"] == "1").sum() == 21
    dropped = pl.read_csv(out / "validate.dropped.tsv", separator="\t", infer_schema=False)
    assert dropped.select("id", "check").rows() == [("9999", "parent_child_count")]


def test_other_commands_do_not_run_it(tmp_path):
    """The limits are validate's flags: summarize reads a large sire."""
    rows = [*_founders(("1", "M"), ("2", "F")), *_kids(60, "2", "1")]
    ped = write_ped(tmp_path / "p.tsv", rows)
    r = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "sum")])
    assert r.returncode == 0, r.stderr
