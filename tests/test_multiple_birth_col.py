"""Behavior tests for --multiple-birth-col, --zygosity-col and the multiple_births section (pedsum#26)."""

from __future__ import annotations

import polars as pl
import pytest
import yaml
from conftest import load_validate_tsv_gz, run_check, run_pedsum, write_ped

from pedsum.base import PedigreeError
from pedsum.checks import CheckStatus
from pedsum.multiple_births import compute_multiple_births, redact_multiple_births
from pedsum.parse import as_zygosity_col
from pedsum.validate import ValidationConfig

PARENTS = [
    {"id": "1", "sex": "F", "mother": "", "father": "", "set": "", "zyg": ""},
    {"id": "2", "sex": "M", "mother": "", "father": "", "set": "", "zyg": ""},
    {"id": "3", "sex": "M", "mother": "", "father": "", "set": "", "zyg": ""},
]
CONFIG = ValidationConfig(multiple_birth_col="set", zygosity_col="zyg")


def _kid(id_: str, sex: str, set_: str, zyg: str = "", mother: str = "1", father: str = "2") -> dict:
    return {"id": id_, "sex": sex, "mother": mother, "father": father, "set": set_, "zyg": zyg}


def _details(name: str, rows: list[dict]) -> dict[str, str]:
    """Each Finding of Check ``name``, keyed by the ID it names."""
    result = run_check(name, rows, CONFIG)
    return {f.detail.split(" ", 1)[0].removeprefix("id="): f.detail.split(": ", 1)[1] for f in result.findings}


def test_zygosity_tokens_parse_mz_dz_and_missing():
    """MZ and DZ in any case, and the missing tokens; anything else raises."""
    assert as_zygosity_col(pl.Series(["mz", "DZ", "NA", "", None]), "z").tolist() == [1, 2, 0, 0, 0]
    with pytest.raises(PedigreeError, match="'monozygotic'"):
        as_zygosity_col(pl.Series(["MZ", "monozygotic"]), "z")


def test_unknown_zygosity_token_blocks(tmp_path):
    """zygosity_tokens is a BLOCK Check."""
    rows = [*PARENTS, _kid("10", "F", "A", "identical"), _kid("11", "F", "A", "MZ")]
    r = _validate(tmp_path, rows)
    assert r.returncode == 2, r.stderr
    assert "zygosity column holds MZ / DZ ......... FAIL" in r.stderr
    assert not (tmp_path / "out" / "validate.tsv.gz").exists()


def test_set_members_that_are_not_full_sibs_are_cleared():
    """A member alone, with an unknown parent, or with other parents than the rest of its set, is cleared."""
    rows = [
        *PARENTS,
        _kid("10", "F", "A"),
        _kid("11", "F", "A"),
        _kid("12", "F", "A", father="3"),  # another father
        _kid("13", "F", "B"),  # alone
        _kid("14", "F", "C", father=""),
        _kid("15", "F", "C", father=""),
    ]
    assert _details("multiple_birth_sets", rows) == {
        "12": "no other member of its set has the same mother and father",
        "13": "it is the only member of its set",
        "14": "a parent is unknown",
        "15": "a parent is unknown",
    }


def test_a_set_holding_two_sibships_is_cleared_whole():
    """Two groups of full sibs under one id: which delivery it means is ambiguous."""
    rows = [
        *PARENTS,
        _kid("10", "F", "A"),
        _kid("11", "F", "A"),
        _kid("12", "F", "A", father="3"),
        _kid("13", "F", "A", father="3"),
    ]
    assert set(_details("multiple_birth_sets", rows)) == {"10", "11", "12", "13"}


def test_mz_marks_that_cannot_form_a_pair_are_cleared():
    """MZ outside a set, alone, three in a set, or co-twins of opposite sex: cleared. A DZ mark in a set stays."""
    rows = [
        *PARENTS,
        _kid("10", "F", "A", "MZ"),
        _kid("11", "F", "A", "MZ"),  # linked
        _kid("12", "F", "B", "MZ"),
        _kid("13", "M", "B", "MZ"),  # opposite sex
        _kid("14", "F", "C", "MZ"),
        _kid("15", "F", "C"),  # only one MZ
        _kid("16", "M", "D", "MZ"),
        _kid("17", "M", "D", "MZ"),
        _kid("18", "M", "D", "MZ"),  # three
        _kid("19", "F", "", "DZ"),  # no set
        _kid("20", "F", "E", "DZ"),
        _kid("21", "M", "E", "DZ"),
    ]
    got = _details("mz_links", rows)
    assert got == {
        "12": "its MZ co-twin id=13 has the opposite sex",
        "13": "its MZ co-twin id=12 has the opposite sex",
        "14": "it is the only MZ member of its set",
        "16": "its set has 3 MZ members; MZ co-twins are linked in pairs",
        "17": "its set has 3 MZ members; MZ co-twins are linked in pairs",
        "18": "its set has 3 MZ members; MZ co-twins are linked in pairs",
        "19": "it is in no multiple-birth set",
    }


def test_mz_links_run_in_the_same_pass_as_a_cleared_set():
    """A cleared set member takes its MZ mark with it, without a second validate."""
    rows = [*PARENTS, _kid("10", "F", "A", "MZ"), _kid("11", "F", "A", "MZ", father="3")]
    assert run_check("mz_links", rows, CONFIG).status == CheckStatus.FAIL
    assert set(_details("mz_links", rows)) == {"10", "11"}


def _validate(tmp_path, rows, *flags, out="out"):
    ped = write_ped(tmp_path / "p.tsv", rows)
    return run_pedsum(
        ["validate", "--in", str(ped), "--out", str(tmp_path / out), "--multiple-birth-col", "set",
         "--zygosity-col", "zyg", *flags]
    )  # fmt: skip


def test_validate_blanks_the_cleared_cells_and_its_output_passes(tmp_path):
    """validate.tsv.gz has the cleared cells blank, and validating it again finds nothing."""
    rows = [*PARENTS, _kid("10", "F", "A", "MZ"), _kid("11", "F", "A", "MZ"), _kid("12", "M", "B", "MZ")]
    r = _validate(tmp_path, rows)
    assert r.returncode == 1, r.stderr
    out = load_validate_tsv_gz(tmp_path / "out")
    assert out.filter(pl.col("id") == "12").select("set", "zyg").row(0) == (None, None)
    assert out.filter(pl.col("id") == "10").select("set", "zyg").row(0) == ("A", "MZ")
    again = run_pedsum(
        ["validate", "--in", str(tmp_path / "out" / "validate.tsv.gz"), "--out", str(tmp_path / "again"),
         "--multiple-birth-col", "set", "--zygosity-col", "zyg"]
    )  # fmt: skip
    assert again.returncode == 0, again.stderr


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--zygosity-col", "zyg"], "--zygosity-col needs --multiple-birth-col"),
        (
            ["--multiple-birth-col", "set", "--birth-date-col", "d", "--twin-max-days", "1"],
            "--twin-max-days and --multiple-birth-col both define the multiple-birth sets",
        ),
    ],
)
def test_flag_combinations_that_are_usage_errors(tmp_path, flags, message):
    """Zygosity needs given sets; given and computed sets exclude each other."""
    ped = write_ped(tmp_path / "p.tsv", PARENTS)
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "out"), *flags])
    assert r.returncode == 2
    assert message in r.stderr


def _summarize(tmp_path, ped, *flags):
    return run_pedsum(
        ["summarize", "--in", str(ped), "--out", str(tmp_path / "sum"), "--multiple-birth-col", "set", *flags]
    )


def _section(tmp_path, name="summary.yaml"):
    return yaml.safe_load((tmp_path / "sum" / name).read_text())["pedigree"]["demography"]["multiple_births"]


def _pairs(tmp_path):
    return yaml.safe_load((tmp_path / "sum" / "summary.yaml").read_text())["pedigree"]["relatedness"][
        "relationship_pairs"
    ]


def test_summarize_refuses_marks_validate_would_clear(tmp_path):
    """An unvalidated annotation stops the fail-fast summarize, as any other finding does."""
    ped = write_ped(tmp_path / "p.tsv", [*PARENTS, _kid("10", "F", "A"), _kid("11", "F", "B")])
    r = _summarize(tmp_path, ped)
    assert r.returncode == 1
    assert "multiple_birth_sets" in r.stderr


def test_zygosity_moves_linked_co_twins_from_fs_to_mz(tmp_path):
    """With --zygosity-col, the MZ co-twins are an MZ pair; without it, all full sibs are FS."""
    rows = [*PARENTS, _kid("10", "F", "A", "MZ"), _kid("11", "F", "A", "MZ"), _kid("12", "M", "")]
    ped = write_ped(tmp_path / "p.tsv", rows)
    assert _summarize(tmp_path, ped).returncode == 0
    assert (_pairs(tmp_path)["FS"], _pairs(tmp_path)["MZ"]) == (3, 0)
    assert _summarize(tmp_path, ped, "--zygosity-col", "zyg").returncode == 0
    assert (_pairs(tmp_path)["FS"], _pairs(tmp_path)["MZ"]) == (2, 1)
    assert _section(tmp_path)["mz_linked_pairs"] == 1


def test_section_counts_sets_pairs_and_weinberg(tmp_path):
    """Sets by size, pairs by sex, the full-sib share, Weinberg on sets of two; decades go to extra."""
    rows = [
        *PARENTS,
        _kid("10", "F", "A"),
        _kid("11", "F", "A"),  # same-sex twins
        _kid("12", "F", "B"),
        _kid("13", "M", "B"),  # opposite-sex twins
        _kid("14", "M", "C"),
        _kid("15", "M", "C"),
        _kid("16", "F", "C"),  # triplets
        _kid("17", "F", ""),
    ]
    for r, year in zip(rows, ["1950"] * 3 + ["1980"] * 2 + ["1990"] * 6, strict=True):
        r["birth_year"] = year
    ped = write_ped(tmp_path / "p.tsv", rows)
    r = _summarize(tmp_path, ped, "--birth-year-col", "birth_year")
    assert r.returncode == 0, r.stderr
    section = _section(tmp_path)
    assert {k: section[k] for k in ("n_sets", "n_individuals", "n_pairs", "sets_by_size")} == {
        "n_sets": 3,
        "n_individuals": 7,
        "n_pairs": 5,
        "sets_by_size": {"2": 2, "3": 1},
    }
    assert section["pairs_by_sex"] == {"male_male": 1, "female_female": 1, "male_female": 3, "unknown_sex": 0}
    assert section["full_sib_pair_fraction"] == pytest.approx(5 / 28, abs=1e-4)  # C(8, 2) full-sib pairs
    assert section["weinberg"] == {
        "n_twin_pairs": 2,
        "same_sex": 1,
        "opposite_sex": 1,
        "mz_pairs_estimate": 0,
        "dz_pairs_estimate": 2,
        "mz_fraction_estimate": 0.0,
    }
    assert "by_birth_decade" not in section
    assert _section(tmp_path, "summary.extra.yaml")["by_birth_decade"] == [
        {"decade": 1980, "n": 1, "male_male": 0, "female_female": 1, "male_female": 0, "unknown_sex": 0},
        {"decade": 1990, "n": 4, "male_male": 1, "female_female": 0, "male_female": 3, "unknown_sex": 0},
    ]


def test_section_absent_without_the_flag(tmp_path):
    """No --multiple-birth-col, no section and no long-TSV row."""
    ped = write_ped(tmp_path / "p.tsv", [*PARENTS, _kid("10", "F", "A"), _kid("11", "F", "A")])
    r = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "sum"), "--tsv"])
    assert r.returncode == 0, r.stderr
    demography = yaml.safe_load((tmp_path / "sum" / "summary.yaml").read_text())["pedigree"]["demography"]
    assert "multiple_births" not in demography
    assert "multiple_births" not in (tmp_path / "sum" / "summary.pedigree.tsv").read_text()


def test_twin_max_days_output_feeds_summarize(tmp_path):
    """The multiple_birth_set column validate --twin-max-days writes is what summarize --multiple-birth-col reads."""
    rows = [
        {"id": "1", "sex": "F", "mother": "", "father": "", "bd": "1960-01-01"},
        {"id": "2", "sex": "M", "mother": "", "father": "", "bd": "1960-01-01"},
        {"id": "10", "sex": "F", "mother": "1", "father": "2", "bd": "1985-06-01"},
        {"id": "11", "sex": "M", "mother": "1", "father": "2", "bd": "1985-06-01"},
    ]
    ped = write_ped(tmp_path / "p.tsv", rows)
    out = tmp_path / "out"
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(out), "--birth-date-col", "bd", "--twin-max-days", "0"])
    assert r.returncode == 0, r.stderr
    r = run_pedsum(
        ["summarize", "--in", str(out / "validate.tsv.gz"), "--out", str(tmp_path / "sum"),
         "--multiple-birth-col", "multiple_birth_set"]
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert _section(tmp_path)["weinberg"]["opposite_sex"] == 1


def test_validating_with_the_column_named_keeps_multiple_birth_set(tmp_path):
    """--multiple-birth-col multiple_birth_set keeps the column that --twin-max-days would otherwise own."""
    rows = [{**r, "multiple_birth_set": r.pop("set")} for r in [*PARENTS, _kid("10", "F", "1"), _kid("11", "F", "1")]]
    ped = write_ped(tmp_path / "p.tsv", rows)
    r = run_pedsum(["validate", "--in", str(ped), "--out", str(tmp_path / "out"), "--multiple-birth-col",
                    "multiple_birth_set"])  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "dropped" not in r.stderr
    assert load_validate_tsv_gz(tmp_path / "out")["multiple_birth_set"].to_list()[-2:] == ["1", "1"]


def _payload(n_same: int, n_opposite: int) -> dict:
    sexes = [("F", "F")] * n_same + [("F", "M")] * n_opposite
    rows = []
    for k, (a, b) in enumerate(sexes):
        rows += [(2 * k, a, k), (2 * k + 1, b, k)]
    df = pl.DataFrame(
        {
            "id": [r[0] for r in rows],
            "sex": [0 if r[1] == "F" else 1 for r in rows],
            "multiple_birth": [r[2] for r in rows],
        },
        schema={"id": pl.Int64, "sex": pl.Int8, "multiple_birth": pl.Int64},
    )
    return compute_multiple_births(df, {"FS": None})


def test_full_sib_share_is_none_when_full_sibs_were_not_counted():
    """--max-degree 0 leaves FS uncounted; the share is None rather than a division by it."""
    assert _payload(1, 1)["full_sib_pair_fraction"] is None


def test_safe_attempt_nulls_weinberg_on_a_small_count():
    """Either Weinberg count below the cell size nulls both counts and every estimate."""
    section = _payload(9, 3)
    redact_multiple_births(section, 5, 100)
    assert section["weinberg"] == {
        "n_twin_pairs": 12,
        "same_sex": None,
        "opposite_sex": None,
        "mz_pairs_estimate": None,
        "dz_pairs_estimate": None,
        "mz_fraction_estimate": None,
    }
    assert section["pairs_by_sex"]["male_female"] is None
    big = _payload(9, 6)
    redact_multiple_births(big, 5, 100)
    assert big["weinberg"]["mz_pairs_estimate"] == 3
