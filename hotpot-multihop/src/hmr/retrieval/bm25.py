"""BM25 (Okapi) tự cài đặt.

score(d, q) = Σ_t qtf(t) · idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
idf(t)      = ln(1 + (N − df + 0.5) / (df + 0.5))   — luôn dương (biến thể Lucene)
Term lặp trong truy vấn được nhân theo qtf (tương đương k3 → ∞).
"""

from __future__ import annotations

import math
from collections import Counter

from hmr.common.interfaces import Index
from hmr.common.tokenizer import Analyzer
from hmr.retrieval.merge import Contribution, accumulate_scores, top_k


class BM25Retriever:
    """Cài đặt ``hmr.common.interfaces.Retriever``."""

    def __init__(self, index: Index, analyzer: Analyzer, k1: float, b: float) -> None:
        self.index = index
        self.analyzer = analyzer
        self.k1 = k1
        self.b = b
        self._n = index.n_docs()
        avgdl = index.avg_doc_len() or 1.0
        # Tính sẵn mẫu số phụ thuộc độ dài: k1·(1 − b + b·|d|/avgdl)
        self._len_norm = [k1 * (1 - b + b * index.doc_len(d) / avgdl) for d in range(self._n)]

    def idf(self, term: str) -> float:
        df = self.index.df(term)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def _contribution(self, weight: float) -> Contribution:
        k1p1, norm = self.k1 + 1, self._len_norm
        return lambda d, tf: weight * tf * k1p1 / (tf + norm[d])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        qtf = Counter(t for t in self.analyzer(query) if self.index.df(t) > 0)
        terms = [(t, self._contribution(c * self.idf(t))) for t, c in qtf.items()]
        ids, scores = accumulate_scores(self.index, terms)
        return top_k(ids, scores, k)
