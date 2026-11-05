"""Chỉ mục phân tầng (IIR mục 7.2.1) [M2 – Việt].

Postings của mỗi term được chia thành các tầng theo impact giảm dần:
- ``per_term``: tầng i của term t gồm phần ``fractions[i]`` postings tiếp theo của t;
- ``global``: ngưỡng impact chung cho toàn chỉ mục (sát ví dụ ngưỡng tf của IIR) —
  term mà mọi posting đều có impact thấp sẽ rơi hết xuống tầng dưới.
Truy vấn duyệt tầng 1 của mọi term; nếu số ứng viên < ``min_candidates`` (mặc định k)
thì lần lượt thêm tầng tiếp theo.
"""

from __future__ import annotations

import logging
import math
from array import array
from collections.abc import Sequence
from typing import Literal

import numpy as np
from tqdm import tqdm

from hmr.common.interfaces import Index
from hmr.common.tokenizer import Analyzer
from hmr.index.skip import SkipPostings
from hmr.retrieval.pruning import PrunedRetriever, Subset, order_by_impact, sort_by_doc
from hmr.retrieval.weighting import TermWeighting

logger = logging.getLogger(__name__)

TierMode = Literal["per_term", "global"]


class TieredIndex:
    """Các tầng postings; mỗi tầng lưu theo doc_id tăng."""

    def __init__(
        self,
        index: Index,
        weighting: TermWeighting,
        fractions: Sequence[float],
        mode: TierMode = "per_term",
        show_progress: bool = True,
    ) -> None:
        if not fractions or any(f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise ValueError(f"fractions phải dương và có tổng 1, nhận {list(fractions)}")
        if mode not in ("per_term", "global"):
            raise ValueError(f"mode phải là per_term|global, nhận {mode!r}")
        self.fractions = list(fractions)
        self.mode = mode
        self.weighting_name = weighting.name
        self._tiers: list[dict[str, tuple[array, array]]] = [{} for _ in fractions]
        cum = [float(c) for c in np.cumsum(self.fractions)[:-1]]
        terms = list(index.vocabulary())
        self.thresholds: list[float] = []
        if mode == "global":
            # Ngưỡng tầng i: impact ≥ phân vị (1 − tổng fractions[0..i]) trên TOÀN BỘ postings
            impacts = np.fromiter(
                (f(d, tf) for t in terms for f in [weighting.impact_fn(t)] for d, tf in index.postings(t)),
                dtype=np.float64,
            )
            self.thresholds = [float(np.quantile(impacts, 1 - c)) for c in cum]
        for term in tqdm(terms, desc=f"Phân tầng {mode}", unit="term", leave=False, disable=not show_progress):
            impact = weighting.impact_fn(term)
            ids, tfs = order_by_impact(index.postings(term), impact)
            n = len(ids)
            # Postings đã sắp impact giảm dần ⇒ mỗi tầng là một đoạn liên tiếp [bounds[i], bounds[i+1])
            if mode == "per_term":
                cuts = [min(n, math.ceil(c * n - 1e-9)) for c in cum]
            else:
                values = [impact(d, tf) for d, tf in zip(ids, tfs)]
                cuts = [sum(1 for v in values if v >= th) for th in self.thresholds]
            bounds = [0, *cuts, n]
            for i in range(len(self.fractions)):
                lo, hi = bounds[i], bounds[i + 1]
                if hi > lo:
                    s_ids, s_tfs = sort_by_doc(ids[lo:hi], tfs[lo:hi])
                    self._tiers[i][term] = (array("I", s_ids), array("I", s_tfs))
        sizes = [sum(len(v[0]) for v in tier.values()) for tier in self._tiers]
        logger.info("Chỉ mục phân tầng (%s, %s): số postings mỗi tầng %s", mode, self.fractions, sizes)

    @property
    def n_tiers(self) -> int:
        return len(self._tiers)

    def tier(self, i: int, term: str) -> Subset:
        """Postings của ``term`` ở tầng i (sắp theo doc_id); rỗng nếu không có."""
        return self._tiers[i].get(term, (array("I"), array("I")))


class TieredRetriever(PrunedRetriever):
    """Duyệt từ tầng cao xuống, dừng khi đủ ``min_candidates`` ứng viên."""

    def __init__(
        self,
        index: Index,
        analyzer: Analyzer,
        weighting: TermWeighting,
        tiers: TieredIndex,
        skip_table: dict[str, SkipPostings] | None,
        rescore: bool = True,
        min_candidates: int | None = None,
    ) -> None:
        super().__init__(index, analyzer, weighting, skip_table, rescore)
        if tiers.weighting_name != weighting.name:
            raise ValueError(f"Tầng dựng với {tiers.weighting_name}, nhưng truy vấn dùng {weighting.name}")
        self.tiers = tiers
        self.min_candidates = min_candidates
        self.last_tiers_used = 0

    def select(self, terms: Sequence[str], k: int) -> dict[str, Subset]:
        need = self.min_candidates or k
        chosen: dict[str, tuple[list[int], list[int]]] = {t: ([], []) for t in terms}
        seen: set[int] = set()
        used = 0
        for i in range(self.tiers.n_tiers):
            used = i + 1
            for t in terms:
                ids, tfs = self.tiers.tier(i, t)
                chosen[t][0].extend(ids)
                chosen[t][1].extend(tfs)
                seen.update(ids)
            if len(seen) >= need:
                break
        self.last_tiers_used = used
        return {t: sort_by_doc(ids, tfs) for t, (ids, tfs) in chosen.items()}
