import math
from collections import Counter

import pytest

from hmr.index.inverted import InvertedIndex
from hmr.retrieval.bm25 import BM25Retriever
from hmr.retrieval.merge import merge_add, top_k
from hmr.retrieval.tfidf import TfIdfRetriever

DOCS = {0: ["apple", "banana", "apple"], 1: ["banana", "cherry"], 2: ["cherry", "durian", "elder", "fig"]}


def test_merge_add_two_pointers() -> None:
    ids, scores = merge_add([1, 4, 6], [1.0, 1.0, 1.0], [2, 4, 7], [0.5, 0.5, 0.5])
    assert ids == [1, 2, 4, 6, 7]
    assert scores == [1.0, 0.5, 1.5, 1.0, 0.5]
    assert merge_add([], [], [3], [2.0]) == ([3], [2.0])


def test_top_k_ties_break_by_doc_id() -> None:
    assert top_k([5, 2, 9], [1.0, 1.0, 3.0], 2) == [(9, 3.0), (2, 1.0)]


def _bm25_reference(query: list[str], k1: float, b: float) -> dict[int, float]:
    """Tính BM25 trực tiếp theo công thức, không qua chỉ mục."""
    n = len(DOCS)
    avgdl = sum(map(len, DOCS.values())) / n
    scores: dict[int, float] = {}
    for d, toks in DOCS.items():
        tf = Counter(toks)
        s = 0.0
        for t, qtf in Counter(query).items():
            df = sum(t in ts for ts in DOCS.values())
            if tf[t] == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += qtf * idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(toks) / avgdl))
        if s:
            scores[d] = s
    return scores


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (0.9, 0.4), (2.0, 0.0)])
def test_bm25_matches_formula(toy_index: InvertedIndex, k1: float, b: float) -> None:
    query = "banana cherry cherry apple"
    got = dict(BM25Retriever(toy_index, str.split, k1=k1, b=b).search(query, 10))
    ref = _bm25_reference(query.split(), k1, b)
    assert got.keys() == ref.keys()
    for d in ref:
        assert got[d] == pytest.approx(ref[d])


def test_bm25_ranking_and_edge_cases(toy_index: InvertedIndex) -> None:
    bm25 = BM25Retriever(toy_index, str.split, k1=1.2, b=0.75)
    res = bm25.search("apple", 10)
    assert [d for d, _ in res] == [0]
    assert bm25.search("banana cherry", 1)[0][0] == 1  # doc 1 chứa cả hai
    assert bm25.search("unknown words", 10) == []
    scores = [s for _, s in bm25.search("banana cherry fig", 10)]
    assert scores == sorted(scores, reverse=True)


def test_tfidf_lnc_ltc_matches_cosine(toy_index: InvertedIndex) -> None:
    query = "banana cherry cherry"
    n = len(DOCS)
    vocab = sorted({t for ts in DOCS.values() for t in ts})
    df = {t: sum(t in ts for ts in DOCS.values()) for t in vocab}

    def unit(v: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(x * x for x in v.values()))
        return {t: x / norm for t, x in v.items()}

    q = unit({t: (1 + math.log10(c)) * math.log10(n / df[t]) for t, c in Counter(query.split()).items()})
    expected = {}
    for d, toks in DOCS.items():
        dv = unit({t: 1 + math.log10(c) for t, c in Counter(toks).items()})
        s = sum(q[t] * dv.get(t, 0.0) for t in q)
        if s:
            expected[d] = s

    got = dict(TfIdfRetriever(toy_index, str.split).search(query, 10))
    assert got.keys() == expected.keys()
    for d in expected:
        assert got[d] == pytest.approx(expected[d])
    assert TfIdfRetriever(toy_index, str.split).search("nothing", 5) == []
