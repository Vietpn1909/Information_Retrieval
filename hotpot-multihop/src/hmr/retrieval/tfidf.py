"""tf-idf theo sơ đồ SMART lnc.ltc (IIR, mục 6.4.3).

Tài liệu (lnc): w = 1 + log10(tf), không idf, chuẩn hoá cosine.
Truy vấn (ltc): w = (1 + log10(qtf)) · log10(N/df), chuẩn hoá cosine.
score(d, q) = Σ_t w_q(t) · w_d(t)
"""

from __future__ import annotations

import logging
import math
from collections import Counter

from tqdm import tqdm

from hmr.common.interfaces import Index
from hmr.common.tokenizer import Analyzer
from hmr.retrieval.merge import Contribution, accumulate_scores, top_k

logger = logging.getLogger(__name__)


def _log_tf(tf: int) -> float:
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


class TfIdfRetriever:
    """Cài đặt ``hmr.common.interfaces.Retriever``."""

    def __init__(self, index: Index, analyzer: Analyzer) -> None:
        self.index = index
        self.analyzer = analyzer
        self._n = index.n_docs()
        self._doc_norm = self._compute_doc_norms()

    def _compute_doc_norms(self) -> list[float]:
        """Chuẩn L2 của vectơ lnc từng tài liệu — duyệt toàn bộ postings một lần."""
        sq = [0.0] * self._n
        for term in tqdm(list(self.index.vocabulary()), desc="Chuẩn lnc", unit="term", leave=False):
            for d, tf in self.index.postings(term):
                w = _log_tf(tf)
                sq[d] += w * w
        return [math.sqrt(s) for s in sq]

    def query_weights(self, query: str) -> dict[str, float]:
        """Vectơ ltc của truy vấn (đã chuẩn hoá); bỏ term không có trong chỉ mục."""
        qtf = Counter(t for t in self.analyzer(query) if self.index.df(t) > 0)
        w = {t: _log_tf(c) * math.log10(self._n / self.index.df(t)) for t, c in qtf.items()}
        norm = math.sqrt(sum(x * x for x in w.values()))
        return {t: x / norm for t, x in w.items()} if norm > 0 else {}

    def _contribution(self, qw: float) -> Contribution:
        doc_norm = self._doc_norm
        return lambda d, tf: qw * _log_tf(tf) / doc_norm[d]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = [(t, self._contribution(qw)) for t, qw in self.query_weights(query).items()]
        ids, scores = accumulate_scores(self.index, terms)
        return top_k(ids, scores, k)
