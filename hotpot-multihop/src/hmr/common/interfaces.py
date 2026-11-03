"""Hợp đồng (interface) chung giữa 5 thành viên.

Mọi chỉ mục / bộ truy xuất trong repo phải tuân theo các Protocol và dataclass
dưới đây. Muốn đổi chữ ký nào phải thống nhất cả nhóm trước.

Quy ước:
- ``doc_id`` là số nguyên liên tục 0..N-1, do ``scripts/build_corpus.py`` gán.
- ``term`` truyền vào ``Index`` là token ĐÃ qua ``hmr.common.tokenizer.tokenize``.
- ``query``/``question`` truyền vào ``Retriever``/``MultiHopRetriever`` là chuỗi thô;
  bộ truy xuất tự tokenize bằng tokenizer chung.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Index(Protocol):
    """Chỉ mục ngược: term -> postings list."""

    def postings(self, term: str) -> list[tuple[int, int]]:
        """Trả ``[(doc_id, tf)]`` sắp tăng theo doc_id; rỗng nếu term không có."""
        ...

    def doc_len(self, doc_id: int) -> int:
        """Số token đã chỉ mục của tài liệu (sau tokenize)."""
        ...

    def avg_doc_len(self) -> float:
        """Độ dài tài liệu trung bình."""
        ...

    def n_docs(self) -> int:
        """Tổng số tài liệu N."""
        ...

    def df(self, term: str) -> int:
        """Document frequency của term."""
        ...

    def vocabulary(self) -> Iterable[str]:
        """Duyệt toàn bộ term (cần cho chuẩn hoá cosine của lnc).

        Bổ sung so với bản chốt ban đầu: tf-idf lnc cần tính chuẩn vectơ tài liệu.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """Bộ truy xuất một bước (tf-idf, BM25, dense, lai...)."""

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Trả tối đa k cặp ``(doc_id, score)`` sắp giảm dần theo score."""
        ...


@dataclass(slots=True)
class RankedPassage:
    """Một đoạn văn trong danh sách kết quả cuối."""

    doc_id: int
    title: str
    sent_ids: list[int]  # câu được chọn trong đoạn; rỗng = cả đoạn
    rank: int  # bắt đầu từ 1
    score: float


@dataclass(slots=True)
class HopTrace:
    """Vết của một bước truy xuất (phục vụ phân tích lỗi đa bước)."""

    query: str
    retrieved_ids: list[int]
    bridge_entities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    """Kết quả đầy đủ cho một câu hỏi — output chính của hệ thống."""

    qid: str
    ranked_passages: list[RankedPassage]
    hops: list[HopTrace]
    latency_ms: float
    config_name: str

    def as_ranking(self) -> list[tuple[int, float]]:
        """Rút gọn thành ``[(doc_id, score)]`` để ghi TREC run."""
        return [(p.doc_id, p.score) for p in self.ranked_passages]


@runtime_checkable
class MultiHopRetriever(Protocol):
    """Bộ truy xuất đa bước.

    Bộ truy xuất không biết qid: có thể để ``qid=""``, nơi gọi sẽ gán lại.
    """

    def search(self, question: str, k: int) -> RetrievalResult:
        """Truy xuất cho một câu hỏi, trả kết quả kèm vết từng hop."""
        ...
