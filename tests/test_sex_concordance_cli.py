"""Offspring Sex Concordance wiring: CLI flags, slim/extra routing, the long TSV, and safe-attempt redaction."""

from __future__ import annotations

import yaml
from conftest import EXAMPLE, write_ped
from conftest import load_summary_extra_yaml as _load_extra
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run
from sex_concordance_helpers import GROUPINGS

from pedsum.schema import _SEX_CONCORDANCE_SLIM_KEYS

_SECTION = "offspring_sex_concordance"


def _concordant_pedigree(tmp_path, sibship_sizes):
    """Pedigree whose Sibships are perfectly sex-concordant (p is tiny).

    ``sibship_sizes`` gives one size per Sibship, so a caller can arrange the
    ``by_group_size`` rows it wants to see.
    """
    rows = []
    next_id = 1
    for s, size in enumerate(sibship_sizes):
        mother, father = next_id, next_id + 1
        rows.append({"id": mother, "sex": 2, "mother": -1, "father": -1})
        rows.append({"id": father, "sex": 1, "mother": -1, "father": -1})
        next_id += 2
        child_sex = 1 if s % 2 == 0 else 2
        for _ in range(size):
            rows.append({"id": next_id, "sex": child_sex, "mother": mother, "father": father})
            next_id += 1
    return write_ped(tmp_path / "concordant.tsv", rows)


def _two_sibship_pedigree(tmp_path):
    """Eight rows forming exactly two eligible Sibships of two."""
    rows = [
        {"id": 1, "sex": 1, "mother": -1, "father": -1},
        {"id": 2, "sex": 2, "mother": -1, "father": -1},
        {"id": 3, "sex": 1, "mother": -1, "father": -1},
        {"id": 4, "sex": 2, "mother": -1, "father": -1},
        {"id": 5, "sex": 1, "mother": 2, "father": 1},
        {"id": 6, "sex": 2, "mother": 2, "father": 1},
        {"id": 7, "sex": 1, "mother": 4, "father": 3},
        {"id": 8, "sex": 2, "mother": 4, "father": 3},
    ]
    return write_ped(tmp_path / "two_sibships.tsv", rows)


def test_default_run_omits_the_section(tmp_path):
    """The feature is opt-in: a bare ``summarize`` emits nothing for it."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    assert _SECTION not in _load_yaml(out_dir)["pedigree"]["demography"]
    assert _SECTION not in _load_extra(out_dir)["pedigree"].get("demography", {})
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "annotated.tsv.gz",
        "summary.extra.yaml",
        "summary.yaml",
    ]


def test_flag_adds_the_section(tmp_path):
    """``--sex-concordance`` emits ``null_model`` plus all three groupings."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance"])
    assert res.returncode == 0, res.stderr
    section = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]
    assert section["null_model"]["method"] == "fixed_margin_exchangeability"
    assert section["null_model"]["headline_eligibility"] == "sex_source_input"
    assert all(name in section for name in GROUPINGS)


def test_null_model_key_is_not_a_bare_null(tmp_path):
    """The block is ``null_model:``; a bare ``null:`` would parse as a None key."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance"])
    assert res.returncode == 0, res.stderr
    raw = yaml.safe_load((out_dir / "summary.yaml").read_text())
    section = raw["pedigree"]["demography"][_SECTION]
    assert None not in section
    assert "null_model" in section


def test_permutations_flag_implies_enable(tmp_path):
    """``--sex-concordance-permutations`` alone turns the analysis on."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--sex-concordance-permutations",
            "50",
            "--sex-concordance-seed",
            "3",
        ],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    assert slim["p_source"] == "permutation"
    perms = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]["permutations"]
    assert perms["requested"] == 50
    assert perms["completed"] == 50
    assert perms["seed"] == 3
    assert "backend" not in perms


def test_seed_without_permutations_warns(tmp_path):
    """A seed with zero permutations is inert, and says so."""
    out_dir = tmp_path / "out"
    res = _run(
        ["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance", "--sex-concordance-seed", "9"],
    )
    assert res.returncode == 0, res.stderr
    assert "--sex-concordance-seed has no effect" in res.stderr


def test_low_analytical_p_without_permutations_warns(tmp_path):
    """Below p = 0.01 the unpermuted analytical p must not be claimed silently."""
    ped = _concordant_pedigree(tmp_path, [4] * 40)
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(ped), "--out", str(out_dir), "--sex-concordance"])
    assert res.returncode == 0, res.stderr
    assert "--sex-concordance-permutations before claiming this" in res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    assert slim["direction"] == "over_concordant"


def test_no_warning_once_permutations_have_run(tmp_path):
    """The warning is about *unpermuted* claims, so permutations silence it."""
    ped = _concordant_pedigree(tmp_path, [4] * 10)
    out_dir = tmp_path / "out"
    res = _run(
        ["summarize", "--in", str(ped), "--out", str(out_dir), "--sex-concordance-permutations", "100"],
    )
    assert res.returncode == 0, res.stderr
    assert "before claiming this" not in res.stderr


def test_slim_extra_partition_has_no_overlap(tmp_path):
    """Each grouping's leaf keys land in exactly one of slim / extra."""
    out_dir = tmp_path / "out"
    res = _run(
        ["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance-permutations", "20"],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]
    extra = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]
    assert "null_model" in slim
    assert "null_model" not in extra
    for name in GROUPINGS:
        assert set(slim[name]) & set(extra[name]) == set()
        assert set(slim[name]) <= set(_SEX_CONCORDANCE_SLIM_KEYS)
        assert "by_group_size" in extra[name]
        assert "all_resolved" in extra[name]
        assert "z" in extra[name]


def test_slim_yaml_stays_under_budget_with_the_feature_on(tmp_path):
    """Even opted in, the example pedigree's slim YAML stays inside the budget."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance"])
    assert res.returncode == 0, res.stderr
    assert len((out_dir / "summary.yaml").read_text().splitlines()) <= 500


def test_long_tsv_carries_aggregates_but_no_group_identities(tmp_path):
    """The optional TSV gains aggregate rows; no mother/father id is emitted."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance", "--tsv"])
    assert res.returncode == 0, res.stderr
    rows = [
        line.split("\t")
        for line in (out_dir / "summary.pedigree.tsv").read_text().splitlines()[1:]
        if line.startswith(_SECTION)
    ]
    assert rows, "expected aggregate concordance rows in the long TSV"
    assert {r[1] for r in rows} == {"null_model", *GROUPINGS}
    forbidden = ("mother", "father", "id")
    assert not any(any(tok in r[2] for tok in forbidden) for r in rows)


def test_safe_attempt_suppresses_inference_below_five_groups(tmp_path):
    """Under five eligible groups: eligibility metadata stays, inference goes."""
    ped = _two_sibship_pedigree(tmp_path)
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--sex-concordance-permutations",
            "20",
            "--safe-attempt",
        ],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    extra = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]

    assert slim["computed"] is True  # the analysis ran; its results are redacted
    assert slim["excess_concordance"] is None
    assert slim["direction"] is None
    assert slim["p_holm"] is None
    assert slim["max_group_pair_share"] is None
    assert extra["z"] is None
    assert extra["male_proportion_distribution"] is None
    assert extra["by_group_size"] == []
    assert extra["all_resolved"]["excess_concordance"] is None
    # Counts of 1-4 are nulled; the procedure metadata may remain.
    assert slim["n_groups_eligible"] is None
    assert extra["permutations"]["completed"] == 20
    assert extra["permutations"]["p_raw"] is None


def test_safe_attempt_redacts_small_by_group_size_rows(tmp_path):
    """Per-size rows below the cell threshold keep only size and group count."""
    ped = _concordant_pedigree(tmp_path, [3] * 12 + [4] * 2)
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(ped), "--out", str(out_dir), "--sex-concordance", "--safe-attempt"])
    assert res.returncode == 0, res.stderr
    rows = {
        r["group_size"]: r for r in _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]["by_group_size"]
    }
    # 12 Sibships of 3 survive; the 2 Sibships of 4 are a small cell.
    assert rows[3]["n_groups"] == 12
    assert rows[3]["observed_pair_concordance"] is not None
    assert rows[4]["n_groups"] == 2
    assert rows[4]["observed_pair_concordance"] is None
    assert rows[4]["excess_concordance"] is None
    assert rows[4]["n_offspring"] is None


def test_safe_attempt_drops_male_proportion_extrema(tmp_path):
    """The male-proportion distribution loses min/max like every other one."""
    ped = _concordant_pedigree(tmp_path, [3] * 12 + [4] * 2)
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(ped), "--out", str(out_dir), "--sex-concordance", "--safe-attempt"])
    assert res.returncode == 0, res.stderr
    dist = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]["male_proportion_distribution"]
    assert dist is not None
    assert "min" not in dist
    assert "max" not in dist


def test_real_imputation_feeds_the_provenance_tally(tmp_path):
    """The three ``sex_source`` values ``validate`` writes land in the right slots after a round trip.

    Guards a cross-module coupling the fabricated-frame tests cannot: they
    supply ``sex_source`` themselves, so they would still pass if ``validate``
    renamed a provenance — while every real pedigree silently reported zeros.

    The Sibship of six under (1, 2) is built so pedsum's own imputation
    produces all three: two offspring with input sex, two whose sex is missing
    and who are each used as exactly one parent role (``imputed_from_missing``),
    and two whose asserted sex is contradicted by their role
    (``imputed_from_role``).
    """
    rows = [
        {"id": 1, "sex": 2, "mother": -1, "father": -1},
        {"id": 2, "sex": 1, "mother": -1, "father": -1},
        {"id": 3, "sex": 1, "mother": 1, "father": 2},
        {"id": 4, "sex": 2, "mother": 1, "father": 2},
        {"id": 5, "sex": -1, "mother": 1, "father": 2},  # used only as a mother
        {"id": 6, "sex": -1, "mother": 1, "father": 2},  # used only as a father
        {"id": 7, "sex": 1, "mother": 1, "father": 2},  # asserted M, used as a mother
        {"id": 8, "sex": 2, "mother": 1, "father": 2},  # asserted F, used as a father
        {"id": 9, "sex": 1, "mother": -1, "father": -1},
        {"id": 10, "sex": 2, "mother": -1, "father": -1},
        {"id": 11, "sex": 1, "mother": -1, "father": -1},
        {"id": 12, "sex": 2, "mother": -1, "father": -1},
    ]
    next_id = 13
    for mother, father in ((5, 9), (10, 6), (7, 11), (12, 8)):
        for sex in (1, 2):
            rows.append({"id": next_id, "sex": sex, "mother": mother, "father": father})
            next_id += 1
    ped = write_ped(tmp_path / "mixed_provenance.tsv", rows)
    # Two overrides are repairs, so summarize reads validate's Fixed Pedigree,
    # whose sex_source column carries the provenance (ADR 0006).
    res = _run(["validate", "--in", str(ped), "--out", str(tmp_path / "v")])
    assert res.returncode == 1, res.stderr

    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(tmp_path / "v" / "validate.tsv.gz"),
            "--out",
            str(out_dir),
            "--sex-concordance",
            "--no-inbreeding",
        ],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    extra = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]

    # Headline admits only the input-sex offspring; the imputed slots stay zero.
    assert extra["eligible_by_sex_source"] == {"input": 10, "imputed_from_missing": 0, "imputed_from_role": 0}
    assert slim["n_offspring_eligible"] == 10

    sensitivity = extra["all_resolved"]
    assert sensitivity["eligible_by_sex_source"] == {"input": 10, "imputed_from_missing": 2, "imputed_from_role": 2}
    assert sensitivity["n_offspring_eligible"] == 14
    assert sum(sensitivity["eligible_by_sex_source"].values()) == sensitivity["n_offspring_eligible"]


def test_no_input_sex_pedigree_skips_with_its_message(tmp_path):
    """A pedigree whose every resolved sex is imputed refuses the headline."""
    rows = [
        {"id": 1, "sex": -1, "mother": -1, "father": -1},
        {"id": 2, "sex": -1, "mother": -1, "father": -1},
    ]
    rows += [{"id": i, "sex": -1, "mother": 1, "father": 2} for i in range(3, 7)]
    rows += [{"id": i, "sex": -1, "mother": -1, "father": -1} for i in range(7, 11)]
    next_id = 11
    for k, parent in enumerate(range(3, 7)):
        for _ in range(2):
            rows.append({"id": next_id, "sex": -1, "mother": parent, "father": 7 + k})
            next_id += 1
    ped = write_ped(tmp_path / "no_input_sex.tsv", rows)
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--sex-concordance",
            "--allow-missing-sex",
            "--no-inbreeding",
        ],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    extra = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    assert slim["computed"] is False
    assert slim["skip_reason"] == "no_input_sex"
    assert "reproduced" in extra["skip_message"]
