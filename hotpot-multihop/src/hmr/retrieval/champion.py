"""Champion list (IIR mục 7.1.3) [M2 – Việt].

Mỗi term giữ r postings có impact cao nhất (không dùng tf thô: trên corpus này 80%
postings có tf = 1 nên xếp theo tf gần như toàn hoà). Danh sách được lưu theo impact
giảm dần với độ dài ``r_max`` ⇒ champion list cho mọi r ≤ r_max là tiền tố, dựng một lần.
"""

from __future__ import annotations

import heapq
import logging
from array import array
from collections.abc import Sequence

from tqdm import tqdm

from hmr.common.interfaces import Index
from hmr.common.tokenizer import Analyzer
from hmr.index.skip import SkipPostings
from hmr.retrieval.pruning import PrunedRetriever, Subset, order_by_impact, sort_by_doc
from hmr.retrieval.weighting import TermWeighting

logger = logging.getLogger(__name__)


class ChampionLists:
    """Champion list của mọi term, độ dài tối đa ``r_max``, sắp theo impact giảm dần."""

    def __init__(self, index: Index, weighting: TermWeighting, r_max: int, show_progress: bool = True) -> None:
        if r_max < 1:
            raise ValueError("r_max phải ≥ 1")
        self.r_max = r_max
        self.weighting_name = weighting.name
        self._lists: dict[str, tuple[array, array]] = {}
        terms = list(index.vocabulary())
        for term in tqdm(terms, desc=f"Champion r≤{r_max}", unit="term", leave=False, disable=not show_progress):
            postings = index.postings(term)
            impact = weighting.impact_fn(term)
            if len(postings) > r_max:
                postings = heapq.nsmallest(r_max, postings, key=lambda p: (-impact(p[0], p[1]), p[0]))
            ids, tfs = order_by_impact(postings, impact)
            self._lists[term] = (array("I", ids), array("I", tfs))
        logger.info("Champion list: %d term, tổng %d postings", len(self._lists), sum(len(v[0]) for v in self._lists.values()))

    def champions(self, term: str, r: int) -> Subset:
        """r postings impact cao nhất của ``term``, sắp lại theo doc_id."""
        if r > self.r_max:
            raise ValueError(f"r={r} vượt r_max={self.r_max} lúc dựng")
        ids, tfs = self._lists.get(term, (array("I"), array("I")))
        return sort_by_doc(ids[:r], tfs[:r])


class ChampionRetriever(PrunedRetriever):
    """Ứng viên = hợp champion list (r) của các term; mặc định chấm lại chính xác."""

    def __init__(
        self,
        index: Index,
        analyzer: Analyzer,
        weighting: TermWeighting,
        lists: ChampionLists,
        r: int,
        skip_table: dict[str, SkipPostings] | None,
        rescore: bool = True,
    ) -> None:
        super().__init__(index, analyzer, weighting, skip_table, rescore)
        if lists.weighting_name != weighting.name:
            raise ValueError(f"Champion list dựng với {lists.weighting_name}, nhưng truy vấn dùng {weighting.name}")
        if r > lists.r_max:
            raise ValueError(f"r={r} vượt r_max={lists.r_max}")
        self.lists = lists
        self.r = r

    def select(self, terms: Sequence[str], k: int) -> dict[str, Subset]:
        return {t: self.lists.champions(t, self.r) for t in terms}
