"""Pin each Check's FailPolicy and what ``blocking_failures`` derives from it (ADR 0005)."""

from __future__ import annotations

import pytest

from pedsum.checks import CHECKS, DROPPABLE_CHECKS, CheckResult, CheckStatus, FailPolicy, blocking_failures

# The user-facing contract: changing a Check's row here changes whether plain
# validate writes a pedigree and whether --drop-offending can remove the flagged ids.
EXPECTED = {
    "required_columns": FailPolicy.BLOCK,
    "empty_pedigree": FailPolicy.BLOCK,
    "id_dtype": FailPolicy.BLOCK,
    "sex_tokens": FailPolicy.BLOCK,
    "negative_ids": FailPolicy.BLOCK,
    "parent_token_range_mother": FailPolicy.BLOCK,
    "parent_token_range_father": FailPolicy.BLOCK,
    "birth_year_dtype": FailPolicy.BLOCK,
    "birth_date_dtype": FailPolicy.BLOCK,
    "zygosity_tokens": FailPolicy.BLOCK,
    "duplicate_ids": FailPolicy.BLOCK_OR_DROP,
    "parent_refs_sex_conflict": FailPolicy.BLOCK_OR_DROP,
    "sex_role_ambiguity": FailPolicy.BLOCK_OR_DROP,
    "parents_distinct": FailPolicy.BLOCK_OR_DROP,
    "acyclic": FailPolicy.BLOCK_OR_DROP,
    "self_loops": FailPolicy.KEEP_OR_DROP,
    "sex_role_consistency": FailPolicy.KEEP_OR_DROP,
    "birth_year_range": FailPolicy.KEEP_OR_DROP,
    "birth_year_topology": FailPolicy.KEEP_OR_DROP,
    "birth_date_range": FailPolicy.KEEP_OR_DROP,
    "birth_date_year_consistency": FailPolicy.KEEP_OR_DROP,
    "zero_parent_present": FailPolicy.BLOCK,
    "duplicate_rows": FailPolicy.AUTOFIX,
    "cycle_breaks": FailPolicy.AUTOFIX,
    "sex_role_override": FailPolicy.AUTOFIX,
    "parent_refs_present_mother": FailPolicy.AUTOFIX,
    "parent_refs_present_father": FailPolicy.AUTOFIX,
    "multiple_birth_sets": FailPolicy.AUTOFIX,
    "mz_links": FailPolicy.AUTOFIX,
}


def _failing(name: str) -> dict[str, CheckResult]:
    return {
        c.name: CheckResult(name=c.name, status=CheckStatus.FAIL if c.name == name else CheckStatus.PASS)
        for c in CHECKS
    }


def test_every_check_has_its_expected_policy():
    """Every registry Check carries the policy pinned in EXPECTED."""
    assert {c.name: c.fail_policy for c in CHECKS} == EXPECTED


def test_droppable_checks_are_the_drop_policies():
    """DROPPABLE_CHECKS is exactly the block_or_drop and keep_or_drop Checks."""
    assert {n for n, p in EXPECTED.items() if p in (FailPolicy.BLOCK_OR_DROP, FailPolicy.KEEP_OR_DROP)} == (
        DROPPABLE_CHECKS
    )


@pytest.mark.parametrize("name", list(EXPECTED))
@pytest.mark.parametrize("drop_offending", [False, True])
def test_blocking_failures_follows_the_policy(name, drop_offending):
    """A lone FAIL blocks per its policy and the mode."""
    policy = EXPECTED[name]
    blocks = policy is FailPolicy.BLOCK if drop_offending else policy in (FailPolicy.BLOCK, FailPolicy.BLOCK_OR_DROP)
    got = blocking_failures(_failing(name), drop_offending=drop_offending)
    assert [n for n, _msg in got] == ([name] if blocks else [])


def test_blocking_failures_uses_the_block_message_and_registry_order():
    """BLOCKED lines carry the block_message, in registry order."""
    results = {
        c.name: CheckResult(
            name=c.name, status=CheckStatus.FAIL if c.name in ("acyclic", "duplicate_ids") else CheckStatus.PASS
        )
        for c in CHECKS
    }
    assert blocking_failures(results, drop_offending=False) == [
        ("duplicate_ids", "duplicate IDs detected"),
        ("acyclic", "cycle detected"),
    ]


def test_skipped_and_absent_checks_never_block():
    """Only a FAIL blocks; a SKIP or a Check missing from results does not."""
    results = {"required_columns": CheckResult(name="required_columns", status=CheckStatus.FAIL)}
    results["id_dtype"] = CheckResult(name="id_dtype", status=CheckStatus.SKIP, note="required_columns failed")
    assert blocking_failures(results, drop_offending=False) == [("required_columns", "required_columns")]


def test_validation_config_carries_every_option():
    """from_args reads every validation option, so the reduction rounds and the self-verify get them all."""
    from dataclasses import fields

    from pedsum.cli.parser import parse_args
    from pedsum.validate import ValidationConfig

    common = [
        "validate", "--in", "p.tsv", "--out", "o",
        "--id-col", "i", "--sex-col", "s", "--mother-col", "m", "--father-col", "f",
        "--birth-year-col", "b", "--birth-year-min", "1900", "--birth-year-max", "2000", "--missing-parent-token", "0",
        "--min-parent-age", "2", "--birth-date-col", "d",
        "--allow-missing-sex", "--no-override-asserted-sex", "--no-sex-check", "--break-cycles",
    ]  # fmt: skip
    # --twin-max-days and --multiple-birth-col exclude each other, so two command lines cover every option.
    configs = [
        ValidationConfig.from_args(parse_args([*common, *extra]))
        for extra in (["--twin-max-days", "1"], ["--multiple-birth-col", "s2", "--zygosity-col", "z"])
    ]
    assert [f.name for f in fields(ValidationConfig) if all(getattr(c, f.name) == f.default for c in configs)] == []
