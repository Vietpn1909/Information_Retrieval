from hmr.common.interfaces import (
    HopTrace,
    Index,
    MultiHopRetriever,
    RankedPassage,
    RetrievalResult,
    Retriever,
)
from hmr.index.inverted import InvertedIndex
from hmr.retrieval.bm25 import BM25Retriever
from hmr.retrieval.factory import SingleHopAdapter
from hmr.retrieval.tfidf import TfIdfRetriever


def test_implementations_satisfy_protocols(toy_index: InvertedIndex) -> None:
    assert isinstance(toy_index, Index)
    bm25 = BM25Retriever(toy_index, str.split, k1=1.2, b=0.75)
    assert isinstance(bm25, Retriever)
    assert isinstance(TfIdfRetriever(toy_index, str.split), Retriever)
    assert isinstance(SingleHopAdapter(bm25, ["a", "b", "c"], "bm25"), MultiHopRetriever)


def test_retrieval_result_as_ranking() -> None:
    res = RetrievalResult(
        qid="q1",
        ranked_passages=[RankedPassage(7, "T7", [], 1, 2.5), RankedPassage(3, "T3", [0, 2], 2, 1.0)],
        hops=[HopTrace("q", [7, 3])],
        latency_ms=1.0,
        config_name="x",
    )
    assert res.as_ranking() == [(7, 2.5), (3, 1.0)]
    assert res.hops[0].bridge_entities == []


def test_single_hop_adapter_builds_result(toy_index: InvertedIndex) -> None:
    adapter = SingleHopAdapter(BM25Retriever(toy_index, str.split, 1.2, 0.75), ["A", "B", "C"], "bm25")
    res = adapter.search("cherry", 5)
    assert [p.title for p in res.ranked_passages] == ["B", "C"]
    assert [p.rank for p in res.ranked_passages] == [1, 2]
    assert res.hops[0].retrieved_ids == [1, 2]
    assert res.config_name == "bm25" and res.latency_ms >= 0
