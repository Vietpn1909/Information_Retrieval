"""Mô hình trọng số term [M2 – Việt].

Mọi mô hình ở đây có dạng  score(d, q) = Σ_t w_q(t) · impact_t(d):
- ``query_weights``: w_q(t) cho từng term của truy vấn;
- ``impact_fn(term)``: trọng số tài liệu impact_t(d), KHÔNG phụ thuộc truy vấn
  → dùng để xếp hạng postings cho champion list / chỉ mục phân tầng;
- ``contribution(term, qw)``: closure đã gộp ``qw · impact`` cho vòng chấm điểm nóng.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from tqdm import tqdm

from hmr.common.interfaces import Index
from hmr.retrieval.merge import Contribution

# (doc_id, tf) -> trọng số tài liệu của một term
ImpactFn = Callable[[int, int], float]


@runtime_checkable
class TermWeighting(Protocol):
    """Hợp đồng chung cho BM25 và các sơ đồ SMART."""

    name: str

    def query_weights(self, terms: Sequence[str]) -> dict[str, float]:
        """w_q(t); bỏ term không có trong chỉ mục và term có trọng số 0."""
        ...

    def impact_fn(self, term: str) -> ImpactFn:
        """Hàm impact_t(d) của term."""
        ...

    def contribution(self, term: str, qw: float) -> Contribution:
        """Hàm (doc_id, tf) -> qw · impact_t(d)."""
        ...


# --------------------------------------------------------------------- BM25
class BM25Weighting:
    """BM25: w_q = qtf · idf, impact = tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl)).

    idf = ln(1 + (N − df + 0.5)/(df + 0.5)) — luôn dương (biến thể Lucene).
    """

    def __init__(self, index: Index, k1: float, b: float) -> None:
        self.index = index
        self.k1 = k1
        self.b = b
        self.name = f"bm25(k1={k1},b={b})"
        self._n = index.n_docs()
        avgdl = index.avg_doc_len() or 1.0
        # Tính sẵn mẫu số phụ thuộc độ dài: k1·(1 − b + b·|d|/avgdl)
        self._len_norm = [k1 * (1 - b + b * index.doc_len(d) / avgdl) for d in range(self._n)]

    def idf(self, term: str) -> float:
        """idf của term; không phụ thuộc k1, b."""
        df = self.index.df(term)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def query_weights(self, terms: Sequence[str]) -> dict[str, float]:
        """w_q(t) = qtf · idf; bỏ term không có trong chỉ mục."""
        qtf = Counter(t for t in terms if self.index.df(t) > 0)
        return {t: c * self.idf(t) for t, c in qtf.items()}

    def impact_fn(self, term: str) -> ImpactFn:
        """impact = tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))."""
        k1p1, norm = self.k1 + 1, self._len_norm
        return lambda d, tf: tf * k1p1 / (tf + norm[d])

    def contribution(self, term: str, qw: float) -> Contribution:
        """Closure qw · impact, gộp sẵn cho vòng chấm điểm nóng."""
        k1p1, norm = self.k1 + 1, self._len_norm
        return lambda d, tf: qw * tf * k1p1 / (tf + norm[d])


# --------------------------------------------------------------------- SMART
_TF_CODES = "nlabL"
_DF_CODES = "ntp"
_NORM_CODES = "nc"


def parse_smart(scheme: str) -> tuple[str, str]:
    """``"lnc.ltc"`` -> ``("lnc", "ltc")`` (tài liệu, truy vấn); báo lỗi nếu sai ký hiệu."""
    parts = scheme.split(".")
    if len(parts) != 2 or any(len(p) != 3 for p in parts):
        raise ValueError(f"Ký hiệu SMART phải dạng 'ddd.qqq', nhận {scheme!r}")
    for p in parts:
        if p[0] not in _TF_CODES or p[1] not in _DF_CODES or p[2] not in _NORM_CODES:
            raise ValueError(f"Ký hiệu SMART không hỗ trợ: {p!r} (tf∈{_TF_CODES}, df∈{_DF_CODES}, norm∈{_NORM_CODES})")
    return parts[0], parts[1]


def smart_tf(code: str, tf: int, max_tf: int, avg_tf: float) -> float:
    """Thành phần tf (IIR bảng 6.15): n tự nhiên, l log, a augmented, b boolean, L log-average."""
    if tf <= 0:
        return 0.0
    if code == "n":
        return float(tf)
    if code == "l":
        return 1.0 + math.log10(tf)
    if code == "a":
        return 0.5 + 0.5 * tf / max_tf
    if code == "b":
        return 1.0
    return (1.0 + math.log10(tf)) / (1.0 + math.log10(avg_tf))  # "L"


def smart_df(code: str, n_docs: int, df: int) -> float:
    """Thành phần df: n không dùng, t idf = log(N/df), p prob idf = max(0, log((N−df)/df))."""
    if code == "n":
        return 1.0
    if code == "t":
        return math.log10(n_docs / df)
    return max(0.0, math.log10((n_docs - df) / df)) if df < n_docs else 0.0  # "p"


class SmartWeighting:
    """tf-idf theo ký hiệu SMART ``ddd.qqq`` (IIR mục 6.4.3), log cơ số 10."""

    def __init__(self, index: Index, scheme: str = "lnc.ltc", show_progress: bool = True) -> None:
        self.index = index
        self.name = scheme
        self.doc_code, self.query_code = parse_smart(scheme)
        self._n = index.n_docs()
        self._progress = show_progress
        self._max_tf = [1] * self._n
        self._avg_tf = [1.0] * self._n
        if self.doc_code[0] in "aL":
            self._compute_tf_stats()
        self._doc_norm = self._compute_doc_norms() if self.doc_code[2] == "c" else [1.0] * self._n

    def _terms(self, desc: str) -> list[str]:
        terms = list(self.index.vocabulary())
        return list(tqdm(terms, desc=desc, unit="term", leave=False, disable=not self._progress))

    def _compute_tf_stats(self) -> None:
        """max tf và tf trung bình (= |d| / số term phân biệt) của từng tài liệu."""
        uniq = [0] * self._n
        for term in self._terms("Thống kê tf"):
            for d, tf in self.index.postings(term):
                uniq[d] += 1
                if tf > self._max_tf[d]:
                    self._max_tf[d] = tf
        self._avg_tf = [self.index.doc_len(d) / u if u else 1.0 for d, u in enumerate(uniq)]

    def _compute_doc_norms(self) -> list[float]:
        """Chuẩn L2 vectơ tài liệu — duyệt toàn bộ postings một lần."""
        tf_code, df_code = self.doc_code[0], self.doc_code[1]
        sq = [0.0] * self._n
        for term in self._terms(f"Chuẩn {self.doc_code}"):
            dfw = smart_df(df_code, self._n, self.index.df(term))
            for d, tf in self.index.postings(term):
                w = smart_tf(tf_code, tf, self._max_tf[d], self._avg_tf[d]) * dfw
                sq[d] += w * w
        # Tài liệu mọi trọng số bằng 0 thì impact luôn 0; đặt chuẩn 1 để tránh chia cho 0
        return [math.sqrt(s) if s > 0 else 1.0 for s in sq]

    def query_weights(self, terms: Sequence[str]) -> dict[str, float]:
        """Vectơ truy vấn theo ``query_code``; bỏ term không có trong chỉ mục và term trọng số 0."""
        qtf = Counter(t for t in terms if self.index.df(t) > 0)
        if not qtf:
            return {}
        tf_code, df_code, norm_code = self.query_code[0], self.query_code[1], self.query_code[2]
        max_q = max(qtf.values())
        avg_q = sum(qtf.values()) / len(qtf)
        w = {
            t: smart_tf(tf_code, c, max_q, avg_q) * smart_df(df_code, self._n, self.index.df(t))
            for t, c in qtf.items()
        }
        if norm_code == "c":
            norm = math.sqrt(sum(x * x for x in w.values()))
            if norm == 0:
                return {}
            w = {t: x / norm for t, x in w.items()}
        return {t: x for t, x in w.items() if x != 0}

    def impact_fn(self, term: str) -> ImpactFn:
        """Trọng số tài liệu theo ``doc_code``, đã chuẩn hoá."""
        tf_code = self.doc_code[0]
        dfw = smart_df(self.doc_code[1], self._n, self.index.df(term))
        max_tf, avg_tf, norm = self._max_tf, self._avg_tf, self._doc_norm
        return lambda d, tf: smart_tf(tf_code, tf, max_tf[d], avg_tf[d]) * dfw / norm[d]

    def contribution(self, term: str, qw: float) -> Contribution:
        """Closure qw · impact; tf log có nhánh riêng để nhanh hơn."""
        tf_code = self.doc_code[0]
        dfw = smart_df(self.doc_code[1], self._n, self.index.df(term))
        max_tf, avg_tf, norm = self._max_tf, self._avg_tf, self._doc_norm
        if tf_code == "l":  # nhánh nóng của lnc.*: tránh gọi smart_tf
            return lambda d, tf: qw * (1.0 + math.log10(tf)) * dfw / norm[d]
        return lambda d, tf: qw * smart_tf(tf_code, tf, max_tf[d], avg_tf[d]) * dfw / norm[d]
