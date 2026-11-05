"""Dựng bộ truy xuất từ tên cấu hình trong ``retrievers`` của YAML.

- Truy xuất thưa [M2]: ``type: bm25 | tfidf`` (+ ``scheme`` SMART), tuỳ chọn
  ``pruning: {kind: champion | tiered, ...}`` — cài ngay trong file này.
- Các loại khác được ĐĂNG KÝ SẴN trong ``EXTERNAL_BUILDERS``: thành viên chỉ cần tạo
  module tương ứng với hàm ``build(cfg, rcfg)`` — KHÔNG cần sửa file này.
- ``type`` bắt đầu bằng ``multihop`` ⇒ builder trả ``MultiHopRetriever`` (trả ``RetrievalResult``);
  mọi loại khác trả ``Retriever`` và được bọc bằng ``SingleHopAdapter``.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from hmr.common.config import Config, index_path, retriever_config
from hmr.common.interfaces import HopTrace, Index, MultiHopRetriever, RankedPassage, RetrievalResult, Retriever
from hmr.index.inverted import InvertedIndex
from hmr.index.skip import build_skip_table
from hmr.retrieval.champion import ChampionLists, ChampionRetriever
from hmr.retrieval.scoring import ExhaustiveRetriever
from hmr.retrieval.tiered import TieredIndex, TieredRetriever
from hmr.retrieval.weighting import BM25Weighting, SmartWeighting, TermWeighting

logger = logging.getLogger(__name__)

SPARSE_TYPES = ("bm25", "tfidf")

# type -> "module:hàm". Hàm có chữ ký build(cfg: Config, rcfg: Mapping[str, Any]) và trả
# Retriever (hoặc MultiHopRetriever nếu type bắt đầu bằng "multihop"). Import lười khi dùng.
EXTERNAL_BUILDERS: dict[str, str] = {
    "dense": "hmr.retrieval.dense:build",  # [M3 – Khiêm]
    "hybrid": "hmr.retrieval.hybrid:build",  # [M3 – Khiêm]
    "rerank": "hmr.retrieval.rerank:build",  # [M3 – Khiêm]
    "multihop": "hmr.multihop.factory:build",  # [M4 – Cường]
}


def is_multihop(rcfg: Mapping[str, Any]) -> bool:
    """Cấu hình đa bước ⇔ ``type`` bắt đầu bằng ``multihop``."""
    return str(rcfg["type"]).startswith("multihop")


def load_index(cfg: Config, name: str) -> InvertedIndex:
    """Nạp chỉ mục đã dựng bằng ``scripts/build_index.py``."""
    path = index_path(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có chỉ mục {path}; chạy: python scripts/build_index.py --index {name}")
    logger.info("Nạp chỉ mục %s", path)
    return InvertedIndex.load(path)


def build_weighting(index: Index, rcfg: Mapping[str, Any]) -> TermWeighting:
    """Mô hình trọng số theo ``type`` của cấu hình."""
    kind = rcfg["type"]
    if kind == "bm25":
        return BM25Weighting(index, k1=float(rcfg["k1"]), b=float(rcfg["b"]))
    if kind == "tfidf":
        return SmartWeighting(index, rcfg.get("scheme", "lnc.ltc"))
    raise ValueError(f"Loại truy xuất chưa hỗ trợ: {kind}")


def _external_builder(kind: str) -> Any:
    target = EXTERNAL_BUILDERS[kind]
    module_name, func_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise  # lỗi import BÊN TRONG module đã tồn tại — để nguyên traceback
        raise ValueError(
            f"type '{kind}' chưa được cài: tạo {module_name.replace('.', '/')}.py với hàm {func_name}(cfg, rcfg)"
        ) from exc
    return getattr(module, func_name)


def build_retriever(cfg: Config, name: str) -> Any:
    """Tạo bộ truy xuất theo tên cấu hình.

    Trả ``Retriever``, hoặc ``MultiHopRetriever`` nếu cấu hình là đa bước (``is_multihop``).
    Truy xuất thưa: tokenizer truy vấn lấy từ chính chỉ mục để không bao giờ lệch.
    """
    rcfg = retriever_config(cfg, name)
    kind = rcfg["type"]
    if kind in EXTERNAL_BUILDERS or (is_multihop(rcfg) and kind not in SPARSE_TYPES):
        return _external_builder(kind if kind in EXTERNAL_BUILDERS else "multihop")(cfg, rcfg)
    if kind not in SPARSE_TYPES:
        raise ValueError(f"Loại truy xuất chưa hỗ trợ: {kind}. Có: {[*SPARSE_TYPES, *EXTERNAL_BUILDERS]}")
    index = load_index(cfg, rcfg["index"])
    weighting = build_weighting(index, rcfg)
    pruning = rcfg.get("pruning")
    if not pruning:
        return ExhaustiveRetriever(index, index.analyze, weighting)
    rescore = bool(pruning.get("rescore", True))
    skip_table = build_skip_table(index, pruning.get("skip_step", "sqrt")) if rescore else None
    if pruning["kind"] == "champion":
        r = int(pruning["r"])
        lists = ChampionLists(index, weighting, r_max=r)
        return ChampionRetriever(index, index.analyze, weighting, lists, r, skip_table, rescore)
    if pruning["kind"] == "tiered":
        tiers = TieredIndex(index, weighting, pruning["fractions"], pruning.get("mode", "per_term"))
        return TieredRetriever(
            index, index.analyze, weighting, tiers, skip_table, rescore, pruning.get("min_candidates")
        )
    raise ValueError(f"Kiểu cắt tỉa chưa hỗ trợ: {pruning['kind']}")


def build_system(cfg: Config, name: str, titles: Sequence[str]) -> MultiHopRetriever:
    """Hệ thống hoàn chỉnh trả ``RetrievalResult`` — dùng chung cho mọi script chạy truy xuất."""
    system = build_retriever(cfg, name)
    if is_multihop(retriever_config(cfg, name)):
        return system
    return SingleHopAdapter(system, titles, name)


class SingleHopAdapter:
    """Bọc một ``Retriever`` thành ``MultiHopRetriever`` (một hop) để mọi cấu hình
    cùng xuất ``RetrievalResult``."""

    def __init__(self, retriever: Retriever, titles: Sequence[str], config_name: str) -> None:
        self.retriever = retriever
        self.titles = titles
        self.config_name = config_name

    def search(self, question: str, k: int) -> RetrievalResult:
        """Truy xuất một bước, đo latency, đóng gói thành ``RetrievalResult``."""
        t0 = perf_counter()
        ranking = self.retriever.search(question, k)
        latency = (perf_counter() - t0) * 1000
        passages = [
            RankedPassage(doc_id=d, title=self.titles[d], sent_ids=[], rank=r, score=s)
            for r, (d, s) in enumerate(ranking, start=1)
        ]
        hop = HopTrace(query=question, retrieved_ids=[d for d, _ in ranking])
        return RetrievalResult(qid="", ranked_passages=passages, hops=[hop], latency_ms=latency, config_name=self.config_name)
