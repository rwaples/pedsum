"""Property tests for pedsum structural summary primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from pedsum.pedigree_ops import _build_children_csr, _parent_rows
from pedsum.sections import _offspring_dist, compute_sibship_sizes, compute_size_structure

if TYPE_CHECKING:
    import scipy.sparse as sp


@st.composite
def acyclic_pedigree_frames(draw: st.DrawFn, *, max_size: int = 30) -> pd.DataFrame:
    """Generate valid acyclic pedigree frames with parents preceding children."""
    n = draw(st.integers(min_value=0, max_value=max_size))
    mothers: list[int] = []
    fathers: list[int] = []
    depths: list[int] = []
    for i in range(n):
        prior = list(range(i))
        mother = draw(st.sampled_from([-1, *prior])) if prior else -1
        father_choices = [-1, *[p for p in prior if p != mother]]
        father = draw(st.sampled_from(father_choices)) if prior else -1
        mothers.append(mother)
        fathers.append(father)
        parent_depths = [depths[p] for p in (mother, father) if p != -1]
        depths.append((max(parent_depths) + 1) if parent_depths else 0)
    sex = draw(st.lists(st.sampled_from([SEX_UNKNOWN, SEX_FEMALE, SEX_MALE]), min_size=n, max_size=n))
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "sex": np.array(sex, dtype=np.int8),
            "mother": np.array(mothers, dtype=np.int64),
            "father": np.array(fathers, dtype=np.int64),
            "ped_depth": np.array(depths, dtype=np.int32),
        }
    )


def _children_csr(df: pd.DataFrame) -> sp.csr_matrix | None:
    """Build the children CSR for ``df`` using the production row-mapping helper."""
    id_index = pd.Index(df["id"].to_numpy())
    m_row, mask_m = _parent_rows(df["mother"].to_numpy(), id_index)
    f_row, mask_f = _parent_rows(df["father"].to_numpy(), id_index)
    return _build_children_csr(m_row, mask_m, f_row, mask_f, len(df))


@given(acyclic_pedigree_frames())
def test_compute_size_structure_conserves_counts(df: pd.DataFrame) -> None:
    """Size-structure headline counts partition individuals and parent links."""
    summary, labels = compute_size_structure(df, _children_csr(df))
    n = len(df)
    has_mom = df["mother"].to_numpy() != -1
    has_dad = df["father"].to_numpy() != -1

    assert summary["n_total"] == n
    assert len(labels) == n
    assert summary["n_founders"] + summary["n_nonfounders"] == n
    assert summary["n_male"] + summary["n_female"] + summary["n_unknown_sex"] == n
    assert summary["n_mother_links"] == int(has_mom.sum())
    assert summary["n_father_links"] == int(has_dad.sum())
    assert summary["n_parent_child_edges"] == summary["n_mother_links"] + summary["n_father_links"]
    assert summary["n_with_both_parents"] == int((has_mom & has_dad).sum())
    assert summary["n_with_mother_only"] == int((has_mom & ~has_dad).sum())
    assert summary["n_with_father_only"] == int((~has_mom & has_dad).sum())
    assert summary["n_half_founders"] == summary["n_with_mother_only"] + summary["n_with_father_only"]
    assert summary["n_founders"] == int((~has_mom & ~has_dad).sum())
    assert sum(summary["depth_counts"]) == n


@given(acyclic_pedigree_frames(), st.data())
def test_compute_size_structure_is_row_order_invariant(df: pd.DataFrame, data: st.DataObject) -> None:
    """Permuting rows leaves structural summary counts unchanged."""
    summary, _ = compute_size_structure(df, _children_csr(df))
    perm = data.draw(st.permutations(range(len(df))))
    shuffled = df.iloc[list(perm)].reset_index(drop=True)
    shuffled_summary, _ = compute_size_structure(shuffled, _children_csr(shuffled))
    assert shuffled_summary == summary


@given(acyclic_pedigree_frames())
def test_build_children_csr_matches_known_parent_links(df: pd.DataFrame) -> None:
    """The parent→child CSR contains exactly the known parent-child links."""
    csr = _children_csr(df)
    mothers = df["mother"].to_numpy()
    fathers = df["father"].to_numpy()
    n_edges = int((mothers != -1).sum() + (fathers != -1).sum())
    if n_edges == 0:
        assert csr is None
        return

    assert csr is not None
    assert csr.shape == (len(df), len(df))
    assert csr.nnz == n_edges
    dense = csr.toarray()
    id_to_row = {int(i): row for row, i in enumerate(df["id"].to_numpy())}
    for parents in (mothers, fathers):
        for child_row, parent in enumerate(parents):
            if parent != -1:
                assert dense[id_to_row[int(parent)], child_row] == 1


@given(st.lists(st.integers(min_value=0, max_value=20), max_size=100))
def test_offspring_distribution_is_a_pmf(counts: list[int]) -> None:
    """Offspring-count histogram buckets conserve probability mass."""
    hist = _offspring_dist(np.array(counts, dtype=np.int64), len(counts))
    if counts:
        assert sum(hist.values()) == pytest.approx(1.0)
    else:
        assert set(hist.values()) == {0.0}


@given(acyclic_pedigree_frames())
def test_sibship_size_distribution_is_a_pmf(df: pd.DataFrame) -> None:
    """Non-empty sibship-size histograms conserve probability mass over Mating Pairs."""
    summary = compute_sibship_sizes(df)
    both_present = (df["mother"] != -1) & (df["father"] != -1)
    if not both_present.any():
        assert summary == {"empty": True}
        return

    children = df.loc[both_present]
    assert summary["empty"] is False
    assert summary["n_sibships"] == children.groupby(["mother", "father"]).ngroups
    assert sum(summary["size_dist"].values()) == pytest.approx(1.0)
