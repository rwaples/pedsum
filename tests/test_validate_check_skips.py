"""Pin the gated-skip cascade in ``validate_pedigree``.

Each test feeds ``validate_pedigree`` a pedigree that fails one specific
upstream check and asserts the downstream checks SKIP with a non-empty
``skip_reason``. Direct in-process calls so we can inspect the
``CheckResult`` objects (the CLI rolls everything into a stderr summary
which makes precise assertions noisy).
"""

from __future__ import annotations

from conftest import write_ped as _write_ped

import pedigree_summary as ps


def _results_by_name(path) -> dict[str, ps.CheckResult]:
    """Run validate_pedigree and return results indexed by check name."""
    _, results, _, _ = ps.validate_pedigree(path)
    return {r.name: r for r in results}


def test_missing_id_skips_id_dependent_checks(tmp_path):
    """A row with no id → id_dtype FAIL; every check on ids or parents SKIPs."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": None, "sex": "M", "mother": -1, "father": -1},
            {"id": "2", "sex": "F", "mother": -1, "father": -1},
        ],
    )
    results = _results_by_name(ped)
    assert results["id_dtype"].status == "FAIL"
    for name in (
        "negative_ids",
        "duplicate_ids",
        "parent_token_range_mother",
        "parent_refs_present_mother",
        "parent_refs_sex_conflict",
        "sex_role_ambiguity",
        "self_loops",
        "parents_distinct",
        "sex_role_consistency",
        "unknown_sex",
        "acyclic",
    ):
        assert results[name].status == "SKIP", name
        assert results[name].skip_reason, name


def test_string_ids_run_every_check(tmp_path):
    """Non-integer ids parse as strings, and every check runs on them."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": "A1", "sex": "M", "mother": -1, "father": -1},
            {"id": "B", "sex": "F", "mother": -1, "father": -1},
            {"id": "x", "sex": "F", "mother": "B", "father": "A1"},
        ],
    )
    results = _results_by_name(ped)
    assert [n for n, r in results.items() if r.status != "PASS"] == [
        "birth_year_dtype",
        "birth_year_range",
        "birth_year_topology",
    ]


def test_string_mode_still_rejects_negative_integer_ids(tmp_path):
    """A negative integer token is negative in string mode too: ``-1`` as an id could never be a parent."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": "-1", "sex": "F", "mother": -1, "father": -1},
            {"id": "b", "sex": "M", "mother": -1, "father": -1},
            {"id": "c", "sex": "M", "mother": "-5", "father": "b"},
        ],
    )
    results = _results_by_name(ped)
    assert results["negative_ids"].status == "FAIL"
    assert results["parent_token_range_mother"].status == "FAIL"


def test_missing_required_column_returns_with_required_columns_fail(tmp_path):
    """No ``father`` column → required_columns FAIL; every other check SKIP."""
    # Construct a TSV missing the ``father`` column.
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1},
            {"id": 2, "sex": "F", "mother": -1},
        ],
    )
    results = _results_by_name(ped)
    assert results["required_columns"].status == "FAIL"
    # Every non-required-columns check stays SKIP with the cascading reason.
    for name, r in results.items():
        if name == "required_columns":
            continue
        assert r.status == "SKIP", name
        assert r.skip_reason == "required_columns failed", name


def test_bad_sex_token_skips_sex_role_checks(tmp_path):
    """Unknown sex token → sex_tokens FAIL; sex-role checks SKIP."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "X", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
        ],
    )
    results = _results_by_name(ped)
    assert results["sex_tokens"].status == "FAIL"
    for name in ("sex_role_ambiguity", "sex_role_consistency", "unknown_sex"):
        assert results[name].status == "SKIP", name
        assert results[name].skip_reason == "sex_tokens failed", name
