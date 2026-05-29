"""Low-level pedigree array helpers (parent rows, sib groups, depth)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pedsum.base import PedigreeError


def _id_list(ids, max_show: int = 5) -> str:
    ids = list(ids)
    if len(ids) <= max_show:
        return ", ".join(str(i) for i in ids)
    return ", ".join(str(i) for i in ids[:max_show]) + f", ... ({len(ids)} total)"


def _parent_rows(parents: np.ndarray, id_index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Map parent IDs to row indices; -1 for missing. Returns (row_index, present_mask)."""
    out = np.full(len(parents), -1, dtype=np.int64)
    mask = parents != -1
    if mask.any():
        out[mask] = id_index.get_indexer(parents[mask])
    return out, mask


def _full_sib_groups(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    """Per-row full-sib counts plus underlying mating-pair group sizes.

    Returns (fs_count, fs_groups, both_present) where fs_count[i] is the
    number of full sibs of row i (0 if either parent is unknown), fs_groups
    is the groupby-size series indexed by (mother, father), and both_present
    is the boolean row mask for rows with both parents known.
    """
    n = len(df)
    fs_count = np.zeros(n, dtype=np.int64)
    both_present = ((df["mother"] != -1) & (df["father"] != -1)).to_numpy()
    if not both_present.any():
        return fs_count, pd.Series(dtype=np.int64), both_present
    children = df.loc[both_present]
    fs_groups = children.groupby(["mother", "father"]).size()
    idx = pd.MultiIndex.from_arrays(
        [children["mother"].to_numpy(), children["father"].to_numpy()],
        names=["mother", "father"],
    )
    fs_count[np.where(both_present)[0]] = fs_groups.reindex(idx).to_numpy() - 1
    return fs_count, fs_groups, both_present


def _grandparent_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mm, mf, fm, ff) arrays of grandparent IDs (-1 for unknown)."""
    parent_lookup = df.set_index("id")[["mother", "father"]]
    return tuple(
        df[outer].map(parent_lookup[inner]).fillna(-1).astype(np.int64).to_numpy()
        for outer, inner in (
            ("mother", "mother"),
            ("mother", "father"),
            ("father", "mother"),
            ("father", "father"),
        )
    )


def _build_children_csr(
    m_row: np.ndarray,
    mask_m: np.ndarray,
    f_row: np.ndarray,
    mask_f: np.ndarray,
    n: int,
) -> sp.csr_matrix | None:
    """Build parent→child CSR from parent-row arrays. None if no edges."""
    parent_rows = np.concatenate([m_row[mask_m], f_row[mask_f]])
    if len(parent_rows) == 0:
        return None
    child_rows = np.concatenate([np.where(mask_m)[0], np.where(mask_f)[0]])
    return sp.csr_matrix(
        (np.ones(len(parent_rows), dtype=np.int8), (parent_rows, child_rows)),
        shape=(n, n),
    )


def _compute_depth_unordered(
    mother_rows: np.ndarray,
    father_rows: np.ndarray,
    n: int,
) -> np.ndarray:
    """Per-row topological depth, tolerant of any input row order.

    Vectorized fixed-point sweep: founders have depth 0, other rows have
    ``max(parent_depth) + 1``.  Iterates until depths stabilise — unlike
    a single Kahn pass, this does not require parents to already precede
    children in row index.  Returns ``np.int32``.  Raises ``PedigreeError``
    when no progress can be made (true cycle).
    """
    depth = np.full(n, -1, dtype=np.int32)
    depth[(mother_rows < 0) & (father_rows < 0)] = 0
    while True:
        todo = depth < 0
        if not todo.any():
            return depth
        m_safe = np.maximum(mother_rows, 0)
        f_safe = np.maximum(father_rows, 0)
        m_resolved = (mother_rows < 0) | (depth[m_safe] >= 0)
        f_resolved = (father_rows < 0) | (depth[f_safe] >= 0)
        eligible = todo & m_resolved & f_resolved
        if not eligible.any():
            unresolved = np.where(todo)[0][:5].tolist()
            raise PedigreeError(
                f"pedigree contains a cycle: {int(todo.sum())} individual(s) "
                f"could not be topologically ordered (e.g. rows {unresolved})",
            )
        md = np.where(mother_rows >= 0, depth[m_safe], 0)
        fd = np.where(father_rows >= 0, depth[f_safe], 0)
        depth[eligible] = np.maximum(md[eligible], fd[eligible]) + 1
