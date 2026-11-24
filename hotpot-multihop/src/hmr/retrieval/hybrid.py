"""Truy xuất lai [M3 – Khiêm]: gộp danh sách xếp hạng của nhiều bộ truy xuất (BM25 + dense).

Hai cách gộp:
- ``rrf`` — Reciprocal Rank Fusion (Cormack và cộng sự, 2009): ``score(d) = Σ_i w_i / (rrf_k + rank_i(d))``.
  Chỉ dùng thứ hạng nên không cần đưa điểm BM25 và cosine về cùng thang đo.
- ``linear`` — tổ hợp tuyến tính điểm đã chuẩn hoá: ``Σ_i w_i · norm(s_i(d))``, chuẩn hoá ``minmax``
  hoặc ``zscore`` trên chính danh sách ứng viên của từng bộ. Với hai thành phần có thể cho
  ``alpha`` thay cho ``weights``: ``weights = [alpha, 1 - alpha]``.

Đoạn không có trong danh sách của một bộ thì không được cộng gì từ bộ đó (RRF) hoặc nhận
điểm chuẩn hoá thấp nhất của bộ đó (linear).
"""

from __future__ import annotations

import heapq
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from hmr.common.config import Config
from hmr.common.interfaces import Retriever

Ranking = list[tuple[int, float]]


def _top(scores: Mapping[int, float], k: int) -> Ranking:
    """k đoạn điểm cao nhất; hoà điểm thì doc_id nhỏ trước (tất định)."""
    return heapq.nlargest(k, scores.items(), key=lambda p: (p[1], -p[0]))


def rrf_fuse(rankings: Sequence[Ranking], rrf_k: float = 60, weights: Sequence[float] | None = None, k: int = 100) -> Ranking:
    """Reciprocal Rank Fusion; ``rankings[i]`` đã sắp giảm dần, thứ hạng bắt đầu từ 1."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("Số trọng số phải bằng số danh sách xếp hạng")
    fused: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, (doc, _) in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + w / (rrf_k + rank)
    return _top(fused, k)


def normalize(ranking: Ranking, method: str) -> tuple[dict[int, float], float]:
    """Chuẩn hoá điểm một danh sách; trả ``(điểm theo doc, điểm gán cho đoạn vắng mặt)``."""
    if not ranking:
        return {}, 0.0
    scores = [s for _, s in ranking]
    if method == "minmax":
        lo, hi = min(scores), max(scores)
        span = hi - lo
        norm = {d: (s - lo) / span if span > 0 else 1.0 for d, s in ranking}
        return norm, 0.0
    if method == "zscore":
        mu = statistics.fmean(scores)
        sd = statistics.pstdev(scores)
        norm = {d: (s - mu) / sd if sd > 0 else 0.0 for d, s in ranking}
        return norm, min(norm.values())
    raise ValueError(f"Chuẩn hoá chưa hỗ trợ: {method} (minmax | zscore)")


def linear_fuse(rankings: Sequence[Ranking], weights: Sequence[float], norm: str = "minmax", k: int = 100) -> Ranking:
    """Tổ hợp tuyến tính điểm đã chuẩn hoá."""
    if len(weights) != len(rankings):
        raise ValueError("Số trọng số phải bằng số danh sách xếp hạng")
    normed = [normalize(r, norm) for r in rankings]
    docs = {d for r in rankings for d, _ in r}
    fused = {d: sum(w * n.get(d, floor) for w, (n, floor) in zip(weights, normed)) for d in docs}
    return _top(fused, k)


def fusion_weights(params: Mapping[str, Any], n: int) -> list[float]:
    """Trọng số từ cấu hình: ``alpha`` (chỉ khi 2 thành phần) hoặc ``weights`` hoặc đều 1."""
    if params.get("fusion") == "linear" and params.get("alpha") is not None:
        if n != 2:
            raise ValueError("alpha chỉ dùng cho đúng 2 thành phần; dùng weights cho trường hợp khác")
        a = float(params["alpha"])
        return [a, 1.0 - a]
    weights = params.get("weights")
    return [float(w) for w in weights] if weights else [1.0] * n


def fuse(rankings: Sequence[Ranking], params: Mapping[str, Any], k: int) -> Ranking:
    """Gộp theo ``params['fusion']`` ∈ {rrf, linear}."""
    weights = fusion_weights(params, len(rankings))
    kind = params.get("fusion", "rrf")
    if kind == "rrf":
        return rrf_fuse(rankings, float(params.get("rrf_k", 60)), weights, k)
    if kind == "linear":
        return linear_fuse(rankings, weights, params.get("norm", "minmax"), k)
    raise ValueError(f"Cách gộp chưa hỗ trợ: {kind} (rrf | linear)")


def batch_search(retriever: Retriever, queries: Sequence[str], k: int) -> list[Ranking]:
    """Dùng ``search_batch`` nếu bộ truy xuất có (dense mã hoá theo lô), không thì lặp ``search``."""
    if hasattr(retriever, "search_batch"):
        return retriever.search_batch(queries, k)
    return [retriever.search(q, k) for q in queries]


class HybridRetriever:
    """Cài đặt ``Retriever``: lấy ``depth`` ứng viên từ mỗi thành phần rồi gộp."""

    def __init__(self, components: Sequence[Retriever], params: Mapping[str, Any], depth: int) -> None:
        if not components:
            raise ValueError("Hybrid cần ít nhất một thành phần")
        self.components = list(components)
        self.params = dict(params)
        self.depth = depth
        fusion_weights(self.params, len(self.components))  # kiểm tra cấu hình sớm

    def search(self, query: str, k: int) -> Ranking:
        """Top-k sau khi gộp."""
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int) -> list[Ranking]:
        """Như ``search`` cho nhiều truy vấn."""
        depth = max(self.depth, k)
        per_comp = [batch_search(c, queries, depth) for c in self.components]
        return [fuse([pc[i] for pc in per_comp], self.params, k) for i in range(len(queries))]


def build(cfg: Config, rcfg: Mapping[str, Any]) -> HybridRetriever:
    """Builder cho ``type: hybrid``; mỗi thành phần là tên một cấu hình trong ``retrievers``."""
    from hmr.retrieval.factory import build_retriever  # import lười: tránh vòng import

    components = [build_retriever(cfg, name) for name in rcfg["components"]]
    return HybridRetriever(components, rcfg, int(rcfg.get("depth", 200)))
