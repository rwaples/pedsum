"""Pin the gated-skip cascade in ``validate_pedigree``.

Each test feeds ``validate_pedigree`` a pedigree that fails one specific
upstream check and asserts the downstream checks SKIP with a non-empty
``skip_reason``. Direct in-process calls so we can inspect the
``CheckResult`` objects (the CLI rolls everything into a stderr summary
which makes precise assertions noisy).
"""

from __future__ import annotations

import pedigree_summary as ps
from conftest import write_ped as _write_ped


def _results_by_name(path) -> dict[str, ps.CheckResult]:
    """Run validate_pedigree and return results indexed by check name."""
    _, results, _, _ = ps.validate_pedigree(path)
    return {r.name: r for r in results}


def test_bad_id_dtype_skips_id_dependent_checks(tmp_path):
    """Non-integer id → id_dtype FAIL; negative_ids / duplicate_ids SKIP."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": "abc", "sex": "M", "mother": -1, "father": -1},
            {"id": "2", "sex": "F", "mother": -1, "father": -1},
        ],
    )
    results = _results_by_name(ped)
    assert results["id_dtype"].status == "FAIL"
    assert results["negative_ids"].status == "SKIP"
    assert results["negative_ids"].skip_reason
    assert results["duplicate_ids"].status == "SKIP"
    assert results["duplicate_ids"].skip_reason


def test_bad_mother_dtype_skips_parent_dependent_checks(tmp_path):
    """Non-integer mother → mother_dtype FAIL; downstream parent checks SKIP."""
    ped = _write_ped(
        tmp_path / "p.tsv",
        [
            {"id": 1, "sex": "M", "mother": "x", "father": -1},
            {"id": 2, "sex": "F", "mother": "-1", "father": -1},
        ],
    )
    results = _results_by_name(ped)
    assert results["mother_dtype"].status == "FAIL"
    # parent_token_range_mother skipped with the local reason; the wider
    # cascade below uses the "or father_dtype" wording because the same
    # else-branch fires when either parent column failed to parse.
    assert results["parent_token_range_mother"].status == "SKIP"
    assert results["parent_refs_present_mother"].status == "SKIP"
    for name in (
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
