"""Offspring Sex Concordance: Holm step-down correction across the groupings."""

from __future__ import annotations

import numpy as np
import pytest
from sex_concordance_helpers import GROUPINGS, offspring_frame, uniform_groups

from pedsum import sex_concordance as sc


def test_holm_pinned_example():
    """Three raw p-values give the textbook step-down result."""
    adjusted = sc.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)  # 3 * 0.01
    assert adjusted["c"] == pytest.approx(0.06)  # max(0.03, 2 * 0.03)
    assert adjusted["b"] == pytest.approx(0.06)  # max(0.06, 1 * 0.04)


def test_holm_is_monotone_and_bounded():
    """Adjusted values never decrease with raw p and never exceed 1."""
    raw = {"a": 0.4, "b": 0.5, "c": 0.9}
    adjusted = sc.holm_adjust(raw)
    ordered = [adjusted[k] for k in sorted(raw, key=raw.get)]
    assert ordered == sorted(ordered)
    assert all(0.0 <= v <= 1.0 for v in adjusted.values())


def test_holm_handles_ties():
    """Tied raw p-values receive the same adjusted value."""
    adjusted = sc.holm_adjust({"a": 0.02, "b": 0.02, "c": 0.5})
    assert adjusted["a"] == adjusted["b"] == pytest.approx(0.06)


def test_holm_single_hypothesis_is_identity():
    """With one available hypothesis the adjustment is a no-op."""
    assert sc.holm_adjust({"a": 0.017}) == {"a": pytest.approx(0.017)}


def test_holm_drops_skipped_analyses_from_the_family():
    """A skipped grouping is not counted in ``m``, so it cannot inflate the others."""
    assert sc.holm_adjust({"a": 0.02, "b": None}) == {"a": pytest.approx(0.02)}


def test_holm_is_across_groupings_not_group_sizes():
    """The family is the computable groupings; ``by_group_size`` is untested."""
    mothers, fathers = uniform_groups(20, 3)
    rng = np.random.default_rng(5)
    out = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, rng.integers(0, 2, 60)))
    computed = [name for name in GROUPINGS if out[name]["computed"]]
    for name in computed:
        block = out[name]
        assert block["p_analytical_holm"] >= block["p_analytical"]
        assert block["p_analytical_holm"] <= len(computed) * block["p_analytical"] + 1e-12
        assert all("p_analytical" not in row for row in block["by_group_size"])
