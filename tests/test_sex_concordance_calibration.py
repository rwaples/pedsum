"""Offspring Sex Concordance: calibration of the analytical test under simulation.

Both regressions guarded here were found by simulation during design review
and neither is visible to a formula test: the reproduction-conditional
provenance artefact, and the far-tail liberality of the normal approximation.
"""

from __future__ import annotations

import math

import numpy as np
from sex_concordance_helpers import offspring_frame, uniform_groups
from sex_concordance_oracles import sample_concordance_numpy

from pedsum import sex_concordance as sc
from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN


def test_null_calibration_with_complete_input_provenance():
    """Under iid sex with complete ``input`` provenance the test holds its size.

    400 groups of 6, 2000 replicates, nominal 5%. Monte-Carlo se is ~0.5pp, so
    the band is wide enough not to flake and tight enough to catch a real
    mis-calibration.
    """
    rng = np.random.default_rng(20260820)
    n_groups, size = 400, 6
    sizes = np.full(n_groups, size, dtype=np.int64)
    starts = np.arange(0, n_groups * size, size)
    n_reject = 0
    n_reps = 2000
    for _ in range(n_reps):
        labels = rng.integers(0, 2, n_groups * size)
        male_counts = np.add.reduceat(labels, starts)
        n_male = int(male_counts.sum())
        moments = sc.analytical_moments(sizes, n_male, n_groups * size - n_male)
        z = (sc.concordant_pairs(male_counts, sizes) - float(moments.expected)) / math.sqrt(float(moments.variance))
        n_reject += abs(z) > 1.96
    assert 0.035 <= n_reject / n_reps <= 0.070


def _selection_replicate(rng, *, group_varying: bool, mixed: bool) -> dict:
    """One replicate where eligibility is conditional on having reproduced.

    True sex is iid at p=0.5 with **no** real concordance. An offspring is
    resolved either because its sex was supplied (``input``) or because it
    reproduced (``imputed_from_role``); the reproduction rate is sex-specific
    and, under ``group_varying``, differs from group to group — the realistic
    case, since retaining a dam's daughters and culling her sons is decided per
    group.
    """
    n_groups, size = 400, 6
    mothers, fathers = uniform_groups(n_groups, size)
    n = n_groups * size
    true_sex = rng.integers(0, 2, n)
    male_rate = rng.uniform(0.1, 0.9, n_groups) if group_varying else np.full(n_groups, 0.5)
    rate = np.where(true_sex == SEX_MALE, np.repeat(male_rate, size), np.repeat(1 - male_rate, size))
    reproduced = rng.random(n) < rate
    has_input = (rng.random(n) < 0.5) if mixed else np.zeros(n, dtype=bool)
    sources = np.where(has_input, "input", np.where(reproduced, "imputed_from_role", "unresolved"))
    sexes = np.where(sources == "unresolved", SEX_UNKNOWN, true_sex)
    return sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, sexes, sources))["sibship"]


def _rejection_rates(group_varying: bool, n_reps: int = 200) -> tuple[float, float]:
    """Return ``(headline, all_resolved)`` rejection rates at nominal 5%."""
    rng = np.random.default_rng(4242)
    headline = all_resolved = 0
    for _ in range(n_reps):
        block = _selection_replicate(rng, group_varying=group_varying, mixed=True)
        if block["computed"] and block["p_analytical"] < 0.05:
            headline += 1
        sens = block["all_resolved"]
        if sens["computed"] and sens["p_analytical"] < 0.05:
            all_resolved += 1
    return headline / n_reps, all_resolved / n_reps


def test_uniform_selection_rates_are_harmless():
    """A *uniform* marginal sex bias in reproduction is absorbed by conditioning.

    This is the case that makes ``sex_source`` counts insufficient on their own:
    it has the same provenance counts as the group-varying case below and the
    opposite calibration.
    """
    headline, all_resolved = _rejection_rates(group_varying=False)
    assert headline < 0.15
    assert all_resolved < 0.15


def test_group_varying_selection_inflates_all_resolved_only():
    """Group-level heterogeneity in selection breaks all-resolved, not the headline."""
    headline, all_resolved = _rejection_rates(group_varying=True)
    assert headline < 0.15, f"input-only headline should stay calibrated, got {headline}"
    assert all_resolved > 0.15, f"all-resolved should inflate, got {all_resolved}"
    assert all_resolved > 3 * headline


def test_all_imputed_pedigree_refuses_the_headline():
    """With no input sex at all the headline skips rather than reporting the artefact."""
    rng = np.random.default_rng(7)
    block = _selection_replicate(rng, group_varying=True, mixed=False)
    assert block["computed"] is False
    assert block["skip_reason"] == "no_input_sex"
    assert "reproduced" in block["skip_message"]


def test_far_tail_is_liberal_for_a_dominant_group():
    """Pin the known far-tail anti-conservatism so an "improvement" can't hide it.

    One group of 600 plus 400 groups of 6 (``max_group_pair_share`` ~0.97).
    At the nominal 0.1% level the analytical p rejects ~1.6% of the time, and
    the excess is entirely in the positive (over-concordant) tail. At the
    conventional 5% level the same configuration is sound — which is why the
    feature warns rather than refusing on dominance grounds.
    """
    sizes = np.concatenate(([600], np.full(400, 6))).astype(np.int64)
    n_total = int(sizes.sum())
    n_male = n_total // 2
    moments = sc.analytical_moments(sizes, n_male, n_total - n_male)
    draws = sample_concordance_numpy(sizes, n_male, 20_000, 1)
    z = (draws - float(moments.expected)) / math.sqrt(float(moments.variance))

    assert 0.04 <= np.mean(np.abs(z) > 1.96) <= 0.07, "should stay sound at the 5% level"
    far_tail = float(np.mean(np.abs(z) > 3.29))
    assert far_tail > 0.005, f"far-tail liberality lost its bite: {far_tail}"
    assert np.mean(z < -3.29) < 0.002, "the excess must be in the positive tail"


def test_max_group_pair_share_tracks_dominance():
    """``max_group_pair_share`` is the diagnostic that predicts the tail behaviour."""
    mothers, fathers = uniform_groups(400, 6)
    flat = sc.compute_offspring_sex_concordance(
        offspring_frame(mothers, fathers, np.resize([SEX_MALE, SEX_FEMALE], mothers.size)),
    )["sibship"]
    assert flat["max_group_pair_share"] < 0.01

    dominant_m = np.concatenate((np.full(600, 10), mothers[: 400 * 6]))
    dominant_f = dominant_m + 1
    dominant = sc.compute_offspring_sex_concordance(
        offspring_frame(dominant_m, dominant_f, np.resize([SEX_MALE, SEX_FEMALE], dominant_m.size)),
    )["sibship"]
    assert dominant["max_group_pair_share"] > 0.9
