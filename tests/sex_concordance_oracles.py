"""An independent NumPy sampler of the Offspring Sex Concordance fixed-margin null, the oracle for the numba sampler."""

from __future__ import annotations

import numpy as np


def sample_concordance_numpy(
    sizes: np.ndarray,
    n_male: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Draw ``n_permutations`` values of ``C`` under the fixed-margin null.

    ``multivariate_hypergeometric(colors=sizes, nsample=M)`` is exactly the
    fixed-margin allocation: draw the ``M`` males from an urn whose colors are
    the groups, ``n_g`` balls each. Its default "marginals" method is the
    sequential conditional-hypergeometric chain, run in C, and documents
    ``sum(colors) < 10**9``.

    Args:
        sizes: Eligible group sizes, in canonical group order.
        n_male: ``M``, held fixed across permutations.
        n_permutations: ``B``, the number of draws.
        seed: Seed for a fresh :class:`numpy.random.Generator`.

    Returns:
        ``(B,)`` int64 array of permuted ``C`` values.
    """
    rng = np.random.default_rng(seed)
    colors = sizes.astype(np.int64, copy=False)
    out = np.empty(n_permutations, dtype=np.int64)
    for b in range(n_permutations):
        male = rng.multivariate_hypergeometric(colors, n_male)
        female = colors - male
        out[b] = ((male * (male - 1) + female * (female - 1)) >> 1).sum()
    return out
