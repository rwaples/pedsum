"""Offspring Sex Concordance: group eligibility, provenance, and membership per grouping."""

from __future__ import annotations

import numpy as np
import pytest
from sex_concordance_helpers import GROUPINGS, offspring_frame, uniform_groups

from pedsum import sex_concordance as sc
from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN


def test_groupings_differ_under_remating():
    """Half-sibs partition differently into Sibships and maternal/paternal groups."""
    # Mother 10 remated: two Sibships of 2, one Maternal Offspring Group of 4,
    # two Paternal Offspring Groups of 2.
    mothers = [10, 10, 10, 10]
    fathers = [11, 11, 12, 12]
    out = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, [1, 0, 1, 0]))
    assert out["sibship"]["n_groups_eligible"] == 2
    assert out["maternal_offspring_group"]["n_groups_eligible"] == 1
    assert out["maternal_offspring_group"]["n_offspring_eligible"] == 4
    assert out["paternal_offspring_group"]["n_groups_eligible"] == 2


def test_missing_parent_changes_eligibility_per_grouping():
    """A missing father removes an offspring from Sibship/paternal, not maternal."""
    mothers = [10, 10, 20, 20]
    fathers = [11, -1, 21, 21]
    out = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, [1, 0, 1, 0]))
    assert out["sibship"]["n_offspring_total"] == 3
    assert out["maternal_offspring_group"]["n_offspring_total"] == 4
    assert out["paternal_offspring_group"]["n_offspring_total"] == 3


def test_eligible_members_retained_when_a_sibling_is_not():
    """An ineligible offspring is dropped; its eligible siblings keep the group."""
    mothers, fathers = uniform_groups(2, 3)
    sources = ["input", "input", "unresolved", "input", "input", "input"]
    sexes = [SEX_MALE, SEX_FEMALE, SEX_UNKNOWN, SEX_MALE, SEX_FEMALE, SEX_MALE]
    block = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, sexes, sources))["sibship"]
    assert block["n_groups_eligible"] == 2
    assert block["n_groups_incomplete"] == 1
    assert block["n_offspring_eligible"] == 5
    assert block["n_offspring_sex_excluded"] == 1


def test_headline_is_input_only_and_sensitivity_admits_imputed():
    """The two passes differ only in which provenances they admit."""
    mothers, fathers = uniform_groups(2, 3)
    sources = ["input", "input", "imputed_from_role", "input", "input", "imputed_from_missing"]
    sexes = [SEX_MALE, SEX_FEMALE, SEX_MALE, SEX_MALE, SEX_FEMALE, SEX_FEMALE]
    block = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, sexes, sources))["sibship"]

    assert block["n_offspring_eligible"] == 4
    assert block["eligible_by_sex_source"] == {"input": 4, "imputed_from_missing": 0, "imputed_from_role": 0}

    sens = block["all_resolved"]
    assert sens["n_offspring_eligible"] == 6
    assert sens["eligible_by_sex_source"] == {"input": 4, "imputed_from_missing": 1, "imputed_from_role": 1}
    # Provenance counts reconcile: every offspring the sensitivity admits is
    # either headline-eligible or one of the imputed categories.
    assert sum(sens["eligible_by_sex_source"].values()) == sens["n_offspring_eligible"]


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_offspring_and_group_counts_conserve(grouping):
    """Total offspring split exactly into excluded / too-small / eligible."""
    rng = np.random.default_rng(3)
    mothers = rng.integers(1, 30, 400) * 10
    fathers = rng.integers(1, 4, 400) + mothers
    sexes = rng.integers(0, 2, 400)
    sources = rng.choice(["input", "imputed_from_role", "unresolved"], 400)
    sexes = np.where(sources == "unresolved", SEX_UNKNOWN, sexes)
    block = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, sexes, sources))[grouping]
    assert (
        block["n_offspring_total"]
        == block["n_offspring_sex_excluded"] + block["n_offspring_in_small_groups"] + block["n_offspring_eligible"]
    )
    assert block["n_groups_total"] == block["n_groups_eligible"] + block["n_groups_too_small"]
    assert block["n_groups_incomplete"] <= block["n_groups_eligible"]


def test_by_group_size_rows_reconcile_with_totals():
    """``by_group_size`` covers every observed eligible size and sums to the totals."""
    mothers = np.concatenate((np.repeat([10, 20], 2), np.repeat([30], 4)))
    fathers = mothers + 1
    block = sc.compute_offspring_sex_concordance(
        offspring_frame(mothers, fathers, [1, 0, 1, 1, 0, 0, 1, 0]),
    )["sibship"]
    rows = block["by_group_size"]
    assert [r["group_size"] for r in rows] == [2, 4]
    assert sum(r["n_groups"] for r in rows) == block["n_groups_eligible"]
    assert sum(r["n_offspring"] for r in rows) == block["n_offspring_eligible"]
    # Every row's expected value is the pooled null, identical across rows.
    assert len({r["expected_pair_concordance"] for r in rows}) == 1
