"""Phần chung của truy xuất có cắt tỉa (champion list, chỉ mục phân tầng) [M2 – Việt].

Khung chung (IIR chương 7):
1. Mỗi term chỉ đóng góp một TẬP CON postings (do lớp con chọn) → tập ứng viên A
   = hợp các tập con (trộn hai con trỏ), kèm điểm xấp xỉ.
2. ``rescore=True``: chấm lại CHÍNH XÁC mọi doc trong A bằng postings đầy đủ, tra tf
   bằng skip pointer. Điểm cộng theo đúng thứ tự term (df tăng dần) như bản vét cạn,
   nên khi tập con = toàn bộ postings thì kết quả trùng vét cạn đến từng bit.
"""

from __future__ import annotations

from collections.abc import Sequence

from hmr.common.interfaces import Index
from hmr.common.tokenizer import Analyzer
from hmr.index.skip import IntersectStats, SkipPostings
from hmr.retrieval.merge import merge_add, top_k
from hmr.retrieval.scoring import SearchStats
from hmr.retrieval.weighting import ImpactFn, TermWeighting

# Tập con postings của một term: (doc_ids tăng dần, tfs tương ứng)
Subset = tuple[Sequence[int], Sequence[int]]


def order_by_impact(postings: Sequence[tuple[int, int]], impact: ImpactFn) -> tuple[list[int], list[int]]:
    """Sắp postings theo impact giảm dần (hoà: doc_id tăng) → (doc_ids, tfs)."""
    ranked = sorted(postings, key=lambda p: (-impact(p[0], p[1]), p[0]))
    return [d for d, _ in ranked], [tf for _, tf in ranked]


def sort_by_doc(doc_ids: Sequence[int], tfs: Sequence[int]) -> Subset:
    """Sắp lại một tập con theo doc_id tăng (để trộn hai con trỏ)."""
    pairs = sorted(zip(doc_ids, tfs))
    return [d for d, _ in pairs], [tf for _, tf in pairs]


class PrunedRetriever:
    """Lớp cơ sở; lớp con cài ``select`` để chọn tập con postings cho từng term."""

    def __init__(
        self,
        index: Index,
        analyzer: Analyzer,
        weighting: TermWeighting,
        skip_table: dict[str, SkipPostings] | None,
        rescore: bool = True,
    ) -> None:
        if rescore and skip_table is None:
            raise ValueError("rescore=True cần skip_table (hmr.index.skip.build_skip_table)")
        self.index = index
        self.analyzer = analyzer
        self.weighting = weighting
        self.skip_table = skip_table
        self.rescore = rescore
        self.last_stats = SearchStats()

    def select(self, terms: Sequence[str], k: int) -> dict[str, Subset]:
        """Tập con postings (sắp theo doc_id) cho từng term — lớp con cài đặt."""
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k trên tập ứng viên đã cắt tỉa; cập nhật ``last_stats``."""
        qw = self.weighting.query_weights(self.analyzer(query))
        terms = sorted(qw, key=self.index.df)  # cùng thứ tự với merge.accumulate_scores
        subsets = self.select(terms, k)
        stats = SearchStats(n_terms=len(terms))

        ids: list[int] = []
        scores: list[float] = []
        for t in terms:
            sub_ids, sub_tfs = subsets[t]
            contrib = self.weighting.contribution(t, qw[t])
            ids, scores = merge_add(ids, scores, sub_ids, [contrib(d, tf) for d, tf in zip(sub_ids, sub_tfs)])
            stats.postings_scored += len(sub_ids)
        stats.candidates = len(ids)

        if self.rescore and self.skip_table is not None:
            skip_stats = IntersectStats()
            exact = [0.0] * len(ids)
            for t in terms:
                sub_ids, sub_tfs = subsets[t]
                if len(sub_ids) == self.index.df(t):  # tập con đã là toàn bộ postings
                    known = dict(zip(sub_ids, sub_tfs))
                    tfs = [known.get(d, 0) for d in ids]
                else:
                    tfs = self.skip_table[t].lookup(ids, skip_stats)
                contrib = self.weighting.contribution(t, qw[t])
                for i, tf in enumerate(tfs):
                    if tf:
                        exact[i] += contrib(ids[i], tf)
            scores = exact
            stats.comparisons = skip_stats.comparisons
        self.last_stats = stats
        return top_k(ids, scores, k)
