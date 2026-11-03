"""Dựng bộ truy xuất từ tên cấu hình trong ``retrievers`` của YAML.

[M3] thêm nhánh ``dense``/``hybrid`` vào ``build_retriever`` khi cài xong.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from time import perf_counter

from hmr.common.config import Config, index_path, retriever_config
from hmr.common.interfaces import HopTrace, RankedPassage, RetrievalResult, Retriever
from hmr.index.inverted import InvertedIndex
from hmr.retrieval.bm25 import BM25Retriever
from hmr.retrieval.tfidf import TfIdfRetriever

logger = logging.getLogger(__name__)


def load_index(cfg: Config, name: str) -> InvertedIndex:
    """Nạp chỉ mục đã dựng bằng ``scripts/build_index.py``."""
    path = index_path(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có chỉ mục {path}; chạy: python scripts/build_index.py --index {name}")
    logger.info("Nạp chỉ mục %s", path)
    return InvertedIndex.load(path)


def build_retriever(cfg: Config, name: str) -> Retriever:
    """Tạo Retriever; tokenizer truy vấn lấy từ chính chỉ mục để không bao giờ lệch."""
    rcfg = retriever_config(cfg, name)
    kind = rcfg["type"]
    if kind == "bm25":
        index = load_index(cfg, rcfg["index"])
        return BM25Retriever(index, index.analyze, k1=float(rcfg["k1"]), b=float(rcfg["b"]))
    if kind == "tfidf":
        index = load_index(cfg, rcfg["index"])
        return TfIdfRetriever(index, index.analyze)
    raise ValueError(f"Loại truy xuất chưa hỗ trợ: {kind}")


class SingleHopAdapter:
    """Bọc một ``Retriever`` thành ``MultiHopRetriever`` (một hop) để mọi cấu hình
    cùng xuất ``RetrievalResult``."""

    def __init__(self, retriever: Retriever, titles: Sequence[str], config_name: str) -> None:
        self.retriever = retriever
        self.titles = titles
        self.config_name = config_name

    def search(self, question: str, k: int) -> RetrievalResult:
        t0 = perf_counter()
        ranking = self.retriever.search(question, k)
        latency = (perf_counter() - t0) * 1000
        passages = [
            RankedPassage(doc_id=d, title=self.titles[d], sent_ids=[], rank=r, score=s)
            for r, (d, s) in enumerate(ranking, start=1)
        ]
        hop = HopTrace(query=question, retrieved_ids=[d for d, _ in ranking])
        return RetrievalResult(qid="", ranked_passages=passages, hops=[hop], latency_ms=latency, config_name=self.config_name)
