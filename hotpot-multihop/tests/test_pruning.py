"""T4–T5: champion list và chỉ mục phân tầng, so với BM25/SMART vét cạn."""

import random

import pytest

from hmr.index.inverted import InvertedIndex
from hmr.index.skip import build_skip_table
from hmr.retrieval.champion import ChampionLists, ChampionRetriever
from hmr.retrieval.scoring import ExhaustiveRetriever
from hmr.retrieval.tiered import TieredIndex, TieredRetriever
from hmr.retrieval.weighting import BM25Weighting, SmartWeighting, TermWeighting

VOCAB = [f"w{i}" for i in range(40)]
ALL = 10**9


@pytest.fixture(scope="module")
def rand_index() -> InvertedIndex:
    """300 tài liệu ngẫu nhiên, phân bố term lệch (Zipf-like) để df chênh nhau nhiều."""
    rng = random.Random(7)
    weights = [1 / (i + 1) for i in range(len(VOCAB))]
    index = InvertedIndex(stem=False, remove_stopwords=False, include_title=False)
    for d in range(300):
        index.add_document(d, rng.choices(VOCAB, weights, k=rng.randint(3, 40)))
    index.finalize()
    return index


@pytest.fixture(scope="module")
def queries() -> list[str]:
    rng = random.Random(8)
    return [" ".join(rng.sample(VOCAB, rng.randint(1, 6))) for _ in range(40)]


def _weightings(index: InvertedIndex) -> list[TermWeighting]:
    return [BM25Weighting(index, 1.2, 0.75), SmartWeighting(index, "lnc.ltc", show_progress=False)]


# ------------------------------------------------------------------ champion
def test_champion_full_r_equals_exhaustive(rand_index: InvertedIndex, queries: list[str]) -> None:
    table = build_skip_table(rand_index, show_progress=False)
    max_df = max(rand_index.df(t) for t in rand_index.vocabulary())
    for w in _weightings(rand_index):
        exhaustive = ExhaustiveRetriever(rand_index, str.split, w)
        lists = ChampionLists(rand_index, w, r_max=max_df, show_progress=False)
        for rescore in (True, False):
            champ = ChampionRetriever(rand_index, str.split, w, lists, max_df, table, rescore)
            for q in queries:
                assert champ.search(q, 20) == exhaustive.search(q, 20)  # trùng từng bit


def test_champion_small_r_rescore_gives_exact_scores(rand_index: InvertedIndex, queries: list[str]) -> None:
    table = build_skip_table(rand_index, show_progress=False)
    w = BM25Weighting(rand_index, 1.2, 0.75)
    exact = ExhaustiveRetriever(rand_index, str.split, w)
    lists = ChampionLists(rand_index, w, r_max=30, show_progress=False)
    resc = ChampionRetriever(rand_index, str.split, w, lists, 10, table, rescore=True)
    approx = ChampionRetriever(rand_index, str.split, w, lists, 10, table, rescore=False)
    pruned_somewhere = False
    for q in queries:
        truth = dict(exact.search(q, ALL))
        got = resc.search(q, 10)
        assert all(s == truth[d] for d, s in got)  # điểm chấm lại = điểm vét cạn
        assert resc.last_stats.candidates <= len(truth)
        pruned_somewhere |= resc.last_stats.candidates < len(truth)
        for d, s in approx.search(q, ALL):
            assert s <= truth[d] + 1e-12  # điểm xấp xỉ chỉ cộng một phần đóng góp (đều dương)
    assert pruned_somewhere


def test_champions_are_top_r_by_impact(rand_index: InvertedIndex) -> None:
    w = BM25Weighting(rand_index, 1.2, 0.75)
    lists = ChampionLists(rand_index, w, r_max=25, show_progress=False)
    for term in rand_index.vocabulary():
        impact = w.impact_fn(term)
        brute = sorted(rand_index.postings(term), key=lambda p: (-impact(*p), p[0]))
        for r in (1, 5, 25):
            ids, tfs = lists.champions(term, r)
            assert list(ids) == sorted(ids)
            assert set(zip(ids, tfs)) == set(brute[:r])


def test_champion_validation(rand_index: InvertedIndex) -> None:
    w = BM25Weighting(rand_index, 1.2, 0.75)
    lists = ChampionLists(rand_index, w, r_max=10, show_progress=False)
    with pytest.raises(ValueError):
        lists.champions("w0", 11)
    with pytest.raises(ValueError):
        ChampionRetriever(rand_index, str.split, w, lists, 5, None, rescore=True)
    with pytest.raises(ValueError):
        ChampionRetriever(rand_index, str.split, BM25Weighting(rand_index, 0.9, 0.4), lists, 5, None, rescore=False)


# ------------------------------------------------------------------ tiered
@pytest.mark.parametrize("mode", ["per_term", "global"])
def test_single_tier_equals_exhaustive(rand_index: InvertedIndex, queries: list[str], mode: str) -> None:
    table = build_skip_table(rand_index, show_progress=False)
    for w in _weightings(rand_index):
        exhaustive = ExhaustiveRetriever(rand_index, str.split, w)
        tiered = TieredRetriever(rand_index, str.split, w, TieredIndex(rand_index, w, [1.0], mode, False), table)
        for q in queries:
            assert tiered.search(q, 20) == exhaustive.search(q, 20)


@pytest.mark.parametrize("mode", ["per_term", "global"])
def test_tiers_partition_postings_by_impact(rand_index: InvertedIndex, mode: str) -> None:
    w = BM25Weighting(rand_index, 1.2, 0.75)
    tiers = TieredIndex(rand_index, w, [0.2, 0.3, 0.5], mode, show_progress=False)
    for term in rand_index.vocabulary():
        impact = w.impact_fn(term)
        parts = [list(zip(*tiers.tier(i, term))) for i in range(3)]
        assert sorted(p for part in parts for p in part) == rand_index.postings(term)  # phân hoạch đủ, không trùng
        for upper, lower in zip(parts, parts[1:]):
            if upper and lower:
                assert min(impact(*p) for p in upper) >= max(impact(*p) for p in lower)
        if mode == "per_term":
            df = rand_index.df(term)
            assert len(parts[0]) == min(df, -(-df * 2 // 10))  # ⌈0.2·df⌉
    if mode == "global":
        assert tiers.thresholds[0] >= tiers.thresholds[1]


def test_tier_fallback(rand_index: InvertedIndex, queries: list[str]) -> None:
    table = build_skip_table(rand_index, show_progress=False)
    w = BM25Weighting(rand_index, 1.2, 0.75)
    tiers = TieredIndex(rand_index, w, [0.1, 0.2, 0.7], "per_term", show_progress=False)
    exhaustive = ExhaustiveRetriever(rand_index, str.split, w)
    only_top = TieredRetriever(rand_index, str.split, w, tiers, table, min_candidates=1)
    everything = TieredRetriever(rand_index, str.split, w, tiers, table, min_candidates=ALL)
    for q in queries:
        only_top.search(q, 10)
        assert only_top.last_tiers_used == 1
        assert everything.search(q, 10) == exhaustive.search(q, 10)
        assert everything.last_tiers_used == 3


def test_tier_validation(rand_index: InvertedIndex) -> None:
    w = BM25Weighting(rand_index, 1.2, 0.75)
    for bad in ([], [0.5, 0.4], [1.2, -0.2]):
        with pytest.raises(ValueError):
            TieredIndex(rand_index, w, bad, show_progress=False)
    with pytest.raises(ValueError):
        TieredIndex(rand_index, w, [1.0], "random", show_progress=False)  # type: ignore[arg-type]
