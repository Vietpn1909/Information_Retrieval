"""BM25 (Okapi) tự cài đặt [M2 – Việt].

score(d, q) = Σ_t qtf(t) · idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
idf(t)      = ln(1 + (N − df + 0.5) / (df + 0.5))   — luôn dương (biến thể Lucene)
Term lặp trong truy vấn được nhân theo qtf (tương đương k3 → ∞).
Công thức nằm ở ``weighting.BM25Weighting``; lớp này là bản duyệt vét cạn.
"""

from __future__ import annotations

from hmr.common.interfaces import Index
from hmr.common.tokenizer import Analyzer
from hmr.retrieval.scoring import ExhaustiveRetriever
from hmr.retrieval.weighting import BM25Weighting


class BM25Retriever(ExhaustiveRetriever):
    """Cài đặt ``hmr.common.interfaces.Retriever``."""

    def __init__(self, index: Index, analyzer: Analyzer, k1: float, b: float) -> None:
        self.bm25 = BM25Weighting(index, k1, b)
        super().__init__(index, analyzer, self.bm25)
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        """idf của term (xem ``BM25Weighting.idf``)."""
        return self.bm25.idf(term)
