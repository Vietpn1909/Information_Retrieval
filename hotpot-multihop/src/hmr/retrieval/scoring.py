"""Truy xuất vét cạn và bộ đếm chi phí truy vấn [M2 – Việt]."""

from __future__ import annotations

from dataclasses import dataclass

from hmr.common.interfaces import Index
from hmr.common.tokenizer import Analyzer
from hmr.retrieval.merge import accumulate_scores, top_k
from hmr.retrieval.weighting import TermWeighting


@dataclass(slots=True)
class SearchStats:
    """Chi phí của lần ``search`` gần nhất — thước đo không phụ thuộc máy (latency Python nhiễu)."""

    n_terms: int = 0
    postings_scored: int = 0  # số posting đã tính điểm
    candidates: int = 0  # số tài liệu được chấm trước khi cắt top-k
    comparisons: int = 0  # phép so sánh doc_id khi tra postings bằng skip pointer


class ExhaustiveRetriever:
    """Duyệt TOÀN BỘ postings của các term truy vấn (term-at-a-time, trộn hai con trỏ).

    Cài đặt ``hmr.common.interfaces.Retriever`` cho mọi ``TermWeighting``.
    """

    def __init__(self, index: Index, analyzer: Analyzer, weighting: TermWeighting) -> None:
        self.index = index
        self.analyzer = analyzer
        self.weighting = weighting
        self.last_stats = SearchStats()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k theo ``weighting``, duyệt toàn bộ postings; cập nhật ``last_stats``."""
        qw = self.weighting.query_weights(self.analyzer(query))
        terms = [(t, self.weighting.contribution(t, w)) for t, w in qw.items()]
        ids, scores = accumulate_scores(self.index, terms)
        self.last_stats = SearchStats(
            n_terms=len(terms), postings_scored=sum(self.index.df(t) for t in qw), candidates=len(ids)
        )
        return top_k(ids, scores, k)
