"""Dò k1/b cho BM25 bằng lưới, chấm vectơ hoá bằng numpy [M2 – Việt].

Mỗi câu hỏi chỉ lấy postings MỘT lần rồi chấm cho mọi (k1, b); điểm được cộng theo
đúng thứ tự term (df tăng dần) và đúng biểu thức của ``BM25Weighting`` nên top-k
trùng với ``BM25Retriever`` (kể cả cách phá hoà điểm: doc_id nhỏ trước).
numpy chỉ dùng cho phép tính mảng; công thức BM25 vẫn tự cài.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from tqdm import tqdm

from hmr.common.interfaces import Index
from hmr.common.io import Qrels, Run
from hmr.common.tokenizer import Analyzer
from hmr.eval.metrics import aggregate, evaluate_per_query
from hmr.retrieval.weighting import BM25Weighting


class QueryPostings(NamedTuple):
    """Postings của một câu hỏi, nối theo thứ tự term df tăng dần."""

    uniq: np.ndarray  # doc_id phân biệt, tăng dần
    inverse: np.ndarray  # vị trí của từng posting trong ``uniq``
    tf: np.ndarray  # float64
    weight: np.ndarray  # qtf·idf của term chứa posting
    dl: np.ndarray  # độ dài tài liệu của posting


@dataclass
class GridResult:
    """Top-``depth`` doc_id cho mọi cấu hình; ``-1`` = không đủ tài liệu có điểm > 0."""

    k1s: list[float]
    bs: list[float]
    qids: list[str]
    depth: int
    top_ids: np.ndarray  # [len(k1s)·len(bs), số câu, depth] int32

    def configs(self) -> list[tuple[float, float]]:
        """Các cặp (k1, b) theo đúng thứ tự chiều đầu của ``top_ids``."""
        return [(k1, b) for k1 in self.k1s for b in self.bs]

    def run(self, cfg_idx: int) -> Run:
        """Run của một cấu hình; score giả = depth − rank để giữ đúng thứ tự hạng khi chấm."""
        run: Run = {}
        for qi, qid in enumerate(self.qids):
            row = self.top_ids[cfg_idx, qi]
            run[qid] = {str(int(d)): float(self.depth - r) for r, d in enumerate(row) if d >= 0}
        return run


def gather(index: Index, weighting: BM25Weighting, terms: Sequence[str], doc_lens: np.ndarray) -> QueryPostings | None:
    """Gom postings của truy vấn; None nếu không term nào có trong chỉ mục."""
    qw = weighting.query_weights(terms)
    if not qw:
        return None
    ids_parts, tf_parts, w_parts = [], [], []
    for term in sorted(qw, key=index.df):  # cùng thứ tự với merge.accumulate_scores
        arr = np.asarray(index.postings(term), dtype=np.int64)
        ids_parts.append(arr[:, 0])
        tf_parts.append(arr[:, 1].astype(np.float64))
        w_parts.append(np.full(len(arr), qw[term]))
    ids = np.concatenate(ids_parts)
    uniq, inverse = np.unique(ids, return_inverse=True)
    return QueryPostings(uniq, inverse, np.concatenate(tf_parts), np.concatenate(w_parts), doc_lens[ids])


def top_ids_for(qp: QueryPostings, k1: float, b: float, avgdl: float, depth: int) -> np.ndarray:
    """Top-depth doc_id cho một (k1, b); điểm > 0, hoà điểm thì doc_id nhỏ trước."""
    norm = k1 * (1 - b + b * qp.dl / avgdl)
    contrib = qp.weight * qp.tf * (k1 + 1) / (qp.tf + norm)
    scores = np.bincount(qp.inverse, weights=contrib, minlength=len(qp.uniq))
    n = len(scores)
    if n > depth:
        kth = np.partition(scores, n - depth)[n - depth]  # điểm lớn thứ depth
        sel = np.nonzero(scores >= kth)[0]
    else:
        sel = np.arange(n)
    sel = sel[scores[sel] > 0]
    order = np.lexsort((qp.uniq[sel], -scores[sel]))[:depth]
    return qp.uniq[sel][order]


def grid_search(
    index: Index,
    analyzer: Analyzer,
    queries: Sequence[tuple[str, str]],
    k1s: Sequence[float],
    bs: Sequence[float],
    depth: int,
    show_progress: bool = True,
) -> GridResult:
    """Chấm toàn bộ lưới (k1, b) cho các ``(qid, câu hỏi)``."""
    weighting = BM25Weighting(index, 1.2, 0.75)  # chỉ dùng idf/qtf — không phụ thuộc k1, b
    doc_lens = np.array([index.doc_len(d) for d in range(index.n_docs())], dtype=np.float64)
    avgdl = index.avg_doc_len() or 1.0
    configs = [(float(k1), float(b)) for k1 in k1s for b in bs]
    top = np.full((len(configs), len(queries), depth), -1, dtype=np.int32)
    for qi, (_, text) in enumerate(tqdm(queries, desc=f"Lưới {len(configs)} cấu hình", unit="q", disable=not show_progress)):
        qp = gather(index, weighting, analyzer(text), doc_lens)
        if qp is None:
            continue
        for ci, (k1, b) in enumerate(configs):
            ids = top_ids_for(qp, k1, b, avgdl, depth)
            top[ci, qi, : len(ids)] = ids
    return GridResult([float(x) for x in k1s], [float(x) for x in bs], [q for q, _ in queries], depth, top)


def evaluate_grid(result: GridResult, qrels: Qrels, measures: Mapping[str, str], show_progress: bool = True) -> list[dict[str, Any]]:
    """Điểm trung bình của từng cấu hình bằng harness chung (``eval.metrics``)."""
    rows: list[dict[str, Any]] = []
    configs = result.configs()
    for ci, (k1, b) in enumerate(tqdm(configs, desc="Chấm lưới", unit="cfg", disable=not show_progress)):
        per_query = evaluate_per_query(result.run(ci), qrels, result.qids, measures)
        _, n, means = aggregate(per_query)[0]
        rows.append({"k1": k1, "b": b, "n": n, **means})
    return rows


def select_best(rows: Sequence[Mapping[str, Any]], objective: str, tiebreak: str) -> Mapping[str, Any]:
    """Cấu hình tốt nhất theo ``objective``, hoà thì theo ``tiebreak``."""
    return max(rows, key=lambda r: (round(r[objective], 10), round(r[tiebreak], 10)))
