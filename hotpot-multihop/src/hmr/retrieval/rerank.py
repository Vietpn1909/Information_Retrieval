"""Xếp hạng lại bằng cross-encoder [M3 – Khiêm].

Bộ truy xuất ``base`` (BM25, dense, hybrid...) sinh ứng viên; cross-encoder đọc **cặp
(câu hỏi, đoạn văn)** cùng lúc và chấm lại ``rerank_depth`` ứng viên đầu. Chính xác hơn
bi-encoder (có attention chéo giữa câu hỏi và đoạn) nhưng chậm — nên chỉ áp lên top-N.

Tuỳ chọn ``combine: {rrf_k, ce_weight}``: gộp RRF thứ hạng cross-encoder với thứ hạng base
thay vì thay thế hẳn (cross-encoder MS MARCO chấm từng đoạn độc lập nên hay đẩy tụt đoạn
hop 2 — xem docs/DENSE.md).

Ứng viên sau vị trí N giữ nguyên thứ tự của ``base`` và luôn đứng sau phần đã chấm lại,
với điểm giảm dần ngặt (để trec_eval không phải tự phá hoà điểm).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hmr.common.config import Config
from hmr.common.interfaces import Retriever
from hmr.common.io import read_jsonl
from hmr.retrieval.dense import passage_text
from hmr.retrieval.hybrid import batch_search, rrf_fuse

Ranking = list[tuple[int, float]]
# Hàm chấm cặp: [(câu hỏi, đoạn văn)] -> điểm liên quan
PairScorer = Callable[[Sequence[tuple[str, str]]], Sequence[float]]

TAIL_STEP = 1e-3  # khoảng cách điểm giữa các ứng viên không được chấm lại


class CrossEncoderScorer:
    """Bọc ``sentence_transformers.CrossEncoder`` thành ``PairScorer``."""

    def __init__(self, name: str, max_length: int, batch_size: int, device: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(name, max_length=max_length, device=device)
        self.batch_size = batch_size

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        if not pairs:
            return []
        scores = self.model.predict([list(p) for p in pairs], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


def order_head(head: Ranking, ce_scores: Sequence[float], combine: Mapping[str, Any] | None = None) -> Ranking:
    """Sắp lại phần đầu theo điểm cross-encoder.

    ``combine = {rrf_k, ce_weight}``: không bỏ hẳn thứ tự ``base`` mà gộp RRF thứ hạng
    cross-encoder (trọng số ``ce_weight``) với thứ hạng base (trọng số 1) — giữ lại tín hiệu
    của base (vd. đoạn hop 2 mà cross-encoder chấm thấp vì ít trùng từ với câu hỏi).
    """
    ce = sorted(zip((d for d, _ in head), (float(s) for s in ce_scores)), key=lambda p: (-p[1], p[0]))
    if not combine:
        return ce
    weights = [float(combine.get("ce_weight", 1.0)), 1.0]
    return rrf_fuse([ce, list(head)], float(combine.get("rrf_k", 60)), weights, k=len(head))


def append_tail(head: Ranking, tail: Ranking) -> Ranking:
    """Nối các ứng viên không được chấm lại phía sau, điểm giảm ngặt dưới phần đầu."""
    floor = min((s for _, s in head), default=0.0)
    return head + [(d, floor - TAIL_STEP * (j + 1)) for j, (d, _) in enumerate(tail)]


def rerank_ranking(
    query: str,
    candidates: Ranking,
    texts: Sequence[str],
    scorer: PairScorer,
    depth: int,
    k: int,
    combine: Mapping[str, Any] | None = None,
) -> Ranking:
    """Chấm lại ``depth`` ứng viên đầu, nối phần còn lại phía sau; trả tối đa k đoạn."""
    head, tail = candidates[:depth], candidates[depth:]
    scores = scorer([(query, texts[d]) for d, _ in head])
    return append_tail(order_head(head, scores, combine), tail)[:k]


class RerankRetriever:
    """Cài đặt ``Retriever``: ``base`` lấy ứng viên → cross-encoder chấm lại top ``depth``."""

    def __init__(
        self,
        base: Retriever,
        scorer: PairScorer,
        texts: Sequence[str],
        depth: int,
        combine: Mapping[str, Any] | None = None,
    ) -> None:
        self.base = base
        self.combine = combine
        self.scorer = scorer
        self.texts = texts
        self.depth = depth

    def search(self, query: str, k: int) -> Ranking:
        """Top-k sau khi xếp hạng lại."""
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int) -> list[Ranking]:
        """Như ``search`` cho nhiều truy vấn (ứng viên lấy theo lô nếu base hỗ trợ)."""
        cands = batch_search(self.base, queries, max(k, self.depth))
        return [rerank_ranking(q, c, self.texts, self.scorer, self.depth, k, self.combine) for q, c in zip(queries, cands)]


def load_passage_texts(cfg: Config) -> list[str]:
    """Văn bản mọi đoạn (đánh chỉ số theo ``doc_id``) theo ``dense.passage_format``."""
    fmt = cfg["dense"]["passage_format"]
    return [passage_text(d, fmt) for d in read_jsonl(cfg["paths"]["corpus"])]


def make_scorer(cfg: Config, key: str) -> CrossEncoderScorer:
    """Cross-encoder theo khoá trong ``dense.rerankers``."""
    rerankers = cfg["dense"]["rerankers"]
    if key not in rerankers:
        raise KeyError(f"Không có reranker '{key}' trong dense.rerankers. Có: {sorted(rerankers)}")
    rc = rerankers[key]
    dcfg = cfg["dense"]
    return CrossEncoderScorer(rc["name"], int(rc["max_length"]), int(dcfg["batch_size"]), dcfg.get("device", "cpu"))


def build(cfg: Config, rcfg: Mapping[str, Any]) -> RerankRetriever:
    """Builder cho ``type: rerank``."""
    from hmr.retrieval.factory import build_retriever  # import lười: tránh vòng import

    base = build_retriever(cfg, rcfg["base"])
    return RerankRetriever(
        base, make_scorer(cfg, rcfg["model"]), load_passage_texts(cfg), int(rcfg["rerank_depth"]), rcfg.get("combine")
    )
