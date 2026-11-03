"""T2: skip pointer — đúng kết quả (so với giao không skip / phép giao tập hợp) và giảm số phép so sánh."""

import random

import pytest

from hmr.index.inverted import InvertedIndex
from hmr.index.skip import (
    IntersectStats,
    SkipPostings,
    build_skip_table,
    intersect,
    intersect_with_skips,
    resolve_step,
    skip_step,
)


def _random_sorted(rng: random.Random, n: int, universe: int) -> list[int]:
    return sorted(rng.sample(range(universe), n))


def test_intersect_two_pointers() -> None:
    stats = IntersectStats()
    assert intersect([1, 3, 5, 7, 9], [2, 3, 4, 9, 10], stats) == [3, 9]
    assert stats.comparisons == 7
    assert intersect([], [1]) == []


def test_iir_example() -> None:
    """IIR hình 2.9: Brutus ∧ Caesar."""
    brutus = [2, 4, 8, 16, 19, 23, 28, 43]
    caesar = [1, 2, 3, 5, 8, 41, 51, 60, 71]
    assert intersect_with_skips(brutus, caesar, 4, 4) == [2, 8]


def test_skip_step() -> None:
    assert [skip_step(n) for n in (0, 1, 4, 99, 100, 25051)] == [1, 1, 2, 9, 10, 158]
    assert resolve_step("sqrt", 100) == 10 and resolve_step(16, 100) == 16 and resolve_step(0, 5) == 1
    with pytest.raises(ValueError):
        resolve_step("log", 10)


@pytest.mark.parametrize("step", [None, 1, 2, 3, 7, 50])
def test_skips_match_plain_intersection(step: int | None) -> None:
    rng = random.Random(0)
    for _ in range(300):
        a = _random_sorted(rng, rng.randint(0, 300), 1000)
        b = _random_sorted(rng, rng.randint(0, 300), 1000)
        expected = sorted(set(a) & set(b))
        assert intersect(a, b) == expected
        assert intersect_with_skips(a, b, step, step) == expected


def test_skips_reduce_comparisons_on_skewed_lists() -> None:
    long_list = list(range(0, 100_000, 2))
    short = [10_000, 50_000, 99_998]
    plain, skip = IntersectStats(), IntersectStats()
    assert intersect(long_list, short, plain) == intersect_with_skips(long_list, short, stats=skip) == short
    assert skip.skips > 0
    assert skip.comparisons < plain.comparisons / 10


def test_lookup_matches_dict() -> None:
    rng = random.Random(1)
    for _ in range(200):
        ids = _random_sorted(rng, rng.randint(0, 400), 2000)
        tfs = [rng.randint(1, 5) for _ in ids]
        cands = _random_sorted(rng, rng.randint(0, 60), 2000)
        truth = dict(zip(ids, tfs))
        for step in (None, 1, 5):
            sp = SkipPostings(ids, tfs, step)
            assert sp.lookup(cands) == [truth.get(c, 0) for c in cands]


def test_lookup_uses_skips() -> None:
    ids = list(range(10_000))
    sp = SkipPostings(ids, [1] * len(ids))
    assert sp.step == 100
    stats = IntersectStats()
    assert sp.lookup([5_000, 9_999, 20_000], stats) == [1, 1, 0]
    assert stats.skips > 0 and stats.comparisons < 500


def test_build_skip_table_mirrors_index(toy_index: InvertedIndex) -> None:
    table = build_skip_table(toy_index, show_progress=False)
    assert table.keys() == set(toy_index.vocabulary())
    for term, sp in table.items():
        assert list(zip(sp.doc_ids, sp.tfs)) == toy_index.postings(term)
