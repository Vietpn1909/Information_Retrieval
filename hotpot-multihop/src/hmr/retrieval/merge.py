"""Trộn điểm trên postings đã sắp xếp (dùng chung cho tf-idf và BM25).

Bộ tích luỹ là hai mảng song song ``(doc_ids, scores)`` sắp tăng theo doc_id.
Mỗi term đóng góp một dãy cũng sắp tăng theo doc_id (lấy từ postings), và được
cộng vào bộ tích luỹ bằng phép trộn hai con trỏ — O(|acc| + |postings|).
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable, Sequence

from hmr.common.interfaces import Index

# (doc_id, tf) -> điểm đóng góp của term cho doc
Contribution = Callable[[int, int], float]


def merge_add(
    a_ids: Sequence[int], a_scores: Sequence[float], b_ids: Sequence[int], b_scores: Sequence[float]
) -> tuple[list[int], list[float]]:
    """Trộn hai dãy (doc_id tăng dần); doc xuất hiện ở cả hai thì cộng điểm."""
    i = j = 0
    na, nb = len(a_ids), len(b_ids)
    out_ids: list[int] = []
    out_scores: list[float] = []
    while i < na and j < nb:
        da, db = a_ids[i], b_ids[j]
        if da == db:
            out_ids.append(da)
            out_scores.append(a_scores[i] + b_scores[j])
            i += 1
            j += 1
        elif da < db:
            out_ids.append(da)
            out_scores.append(a_scores[i])
            i += 1
        else:
            out_ids.append(db)
            out_scores.append(b_scores[j])
            j += 1
    out_ids.extend(a_ids[i:])
    out_scores.extend(a_scores[i:])
    out_ids.extend(b_ids[j:])
    out_scores.extend(b_scores[j:])
    return out_ids, out_scores


def accumulate_scores(index: Index, terms: Iterable[tuple[str, Contribution]]) -> tuple[list[int], list[float]]:
    """Cộng dồn điểm qua các term; xử lý term df nhỏ trước để bộ tích luỹ nhỏ lâu nhất."""
    acc_ids: list[int] = []
    acc_scores: list[float] = []
    for term, contrib in sorted(terms, key=lambda tc: index.df(tc[0])):
        postings = index.postings(term)
        ids = [d for d, _ in postings]
        scores = [contrib(d, tf) for d, tf in postings]
        acc_ids, acc_scores = merge_add(acc_ids, acc_scores, ids, scores)
    return acc_ids, acc_scores


def top_k(ids: Sequence[int], scores: Sequence[float], k: int) -> list[tuple[int, float]]:
    """k cặp điểm cao nhất, giảm dần; hoà điểm thì doc_id nhỏ trước (tất định)."""
    return heapq.nlargest(k, zip(ids, scores), key=lambda p: (p[1], -p[0]))
