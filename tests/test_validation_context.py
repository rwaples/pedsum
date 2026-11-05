"""ValidationContext keeps its derived lookups in step with the arrays they are built from."""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl

from pedsum.pedigree_ops import IdIndex
from pedsum.validate import _CACHES, _ID_CACHES, _ROW_ALIGNED, ValidationConfig, ValidationContext, run_checks


def test_invalidation_lists_name_every_cache_and_row_array():
    """A new cached property or row-aligned field joins these lists, or this test says it did not."""
    assert _CACHES == ("sorted_ids", "id_index", "mother_refs", "father_refs", "depth", "imputation")
    assert set(_ID_CACHES) <= set(_CACHES)
    assert _ROW_ALIGNED == (
        "ids",
        "mothers",
        "fathers",
        "sex",
        "birth_year",
        "birth_date",
        "zygosity",
        "multiple_birth",
        "cleared_multiple_birth",
        "co_twin",
        "cleared_zygosity",
    )


def _arrays(value) -> dict[str, np.ndarray]:
    """A cache's value as named arrays, so two builds compare by content."""
    if isinstance(value, np.ndarray):
        return {"": value}
    if isinstance(value, IdIndex):
        return {"ids": value.to_numpy(), "rows": value.get_indexer(np.arange(-1, 12))}
    return {f.name: np.asarray(getattr(value, f.name)) for f in dataclasses.fields(value)}


def _ctx() -> ValidationContext:
    """A parsed context of three generations, every cache built."""
    rows = [
        ("1", "M", "-1", "-1"),
        ("2", "F", "-1", "-1"),
        ("3", "F", "2", "1"),
        ("4", "M", "3", "1"),
        ("5", "F", "3", "4"),
    ]
    df = pl.DataFrame(rows, schema=dict.fromkeys(("id", "sex", "mother", "father"), pl.String), orient="row")
    ctx = ValidationContext(df, ValidationConfig())
    run_checks(ctx)
    for name in _CACHES:
        getattr(ctx, name)
    return ctx


def _rebuilt(ctx: ValidationContext) -> ValidationContext:
    """A context holding ``ctx``'s arrays and no cache."""
    fresh = ValidationContext(ctx.df_raw, ctx.config)
    for name in _ROW_ALIGNED:
        setattr(fresh, name, getattr(ctx, name))
    return fresh


def _assert_caches_match(ctx: ValidationContext, fresh: ValidationContext) -> None:
    for name in _CACHES:
        got, want = _arrays(getattr(ctx, name)), _arrays(getattr(fresh, name))
        assert got.keys() == want.keys(), name
        for key in got:
            np.testing.assert_array_equal(got[key], want[key], err_msg=f"{name}.{key}")


def test_keep_rows_drops_every_cache():
    """After keep_rows every cache is rebuilt from the kept rows."""
    ctx = _ctx()
    ctx.keep_rows(np.array([True, True, True, False, True]))
    assert not set(_CACHES) & ctx.__dict__.keys()
    assert ctx.ids.tolist() == [1, 2, 3, 5]
    _assert_caches_match(ctx, _rebuilt(ctx))


def test_keep_rows_seeds_sorted_ids():
    """A caller that already sorted the kept ids hands them in, and no sort runs again."""
    ctx = _ctx()
    seed = np.array([1, 2, 3, 5])
    ctx.keep_rows(np.array([True, True, True, False, True]), sorted_ids=seed)
    assert ctx.sorted_ids is seed


def test_clear_parents_keeps_only_the_id_caches():
    """clear_parents leaves the id lookups and rebuilds the parent-derived ones."""
    ctx = _ctx()
    kept = {name: ctx.__dict__[name] for name in _ID_CACHES}
    ctx.clear_parents("mother", np.array([4]))
    for name, value in kept.items():
        assert ctx.__dict__[name] is value
    assert not (set(_CACHES) - set(_ID_CACHES)) & ctx.__dict__.keys()
    assert ctx.mothers[4] == -1
    _assert_caches_match(ctx, _rebuilt(ctx))
