"""tf-idf theo ký hiệu SMART, mặc định lnc.ltc (IIR mục 6.4.3) [M2 – Việt].

Tài liệu (lnc): w = 1 + log10(tf), không idf, chuẩn hoá cosine.
Truy vấn (ltc): w = (1 + log10(qtf)) · log10(N/df), chuẩn hoá cosine.
Các biến thể khác (ltc.ltc, nnc.ntc, anc.atc, ...) xem ``weighting.SmartWeighting``.
"""

from __future__ import annotations

from hmr.common.interfaces import Index
from hmr.common.tokenizer import Analyzer
from hmr.retrieval.scoring import ExhaustiveRetriever
from hmr.retrieval.weighting import SmartWeighting


class TfIdfRetriever(ExhaustiveRetriever):
    """Cài đặt ``hmr.common.interfaces.Retriever``."""

    def __init__(self, index: Index, analyzer: Analyzer, scheme: str = "lnc.ltc") -> None:
        self.smart = SmartWeighting(index, scheme)
        super().__init__(index, analyzer, self.smart)
        self.scheme = scheme
