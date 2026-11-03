"""Chấm điểm run file bằng pytrec_eval, gộp tổng và theo nhóm (type, level).

Câu hỏi có trong tập đánh giá nhưng không có dòng nào trong run được tính 0 điểm
(pytrec_eval mặc định bỏ qua chúng, làm điểm bị thổi phồng).
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import pytrec_eval

from hmr.common.io import Qrels, Run

PerQuery = dict[str, dict[str, float]]

_CUTOFF_RE = re.compile(r"^(.*)_(\d+)$")


def _base_measure(measure: str) -> str:
    """``ndcg_cut_10`` -> ``ndcg_cut``; ``map`` -> ``map``."""
    m = _CUTOFF_RE.match(measure)
    return m.group(1) if m else measure


def evaluate_per_query(run: Run, qrels: Qrels, qids: Iterable[str], measures: Mapping[str, str]) -> PerQuery:
    """Điểm từng câu hỏi: ``{qid: {tên hiển thị: giá trị}}``.

    ``measures``: tên hiển thị -> tên measure pytrec_eval (vd ``{"nDCG@10": "ndcg_cut_10"}``).
    """
    qids = [q for q in qids if q in qrels]
    evaluator = pytrec_eval.RelevanceEvaluator(
        {q: qrels[q] for q in qids}, {_base_measure(m) for m in measures.values()}
    )
    raw = evaluator.evaluate({q: run[q] for q in qids if run.get(q)})
    return {q: {name: raw.get(q, {}).get(m, 0.0) for name, m in measures.items()} for q in qids}


def aggregate(
    per_query: PerQuery,
    groups: Mapping[str, Mapping[str, str]] | None = None,
    group_by: Sequence[str] = (),
) -> list[tuple[str, int, dict[str, float]]]:
    """Trung bình theo nhóm. Trả các dòng ``(nhãn, số câu, {measure: mean})``, dòng đầu là ``all``."""
    buckets: dict[str, list[str]] = defaultdict(list)
    buckets["all"] = list(per_query)
    for key in group_by:
        for q in per_query:
            value = (groups or {}).get(q, {}).get(key)
            if value is not None:
                buckets[f"{key}={value}"].append(q)
    rows: list[tuple[str, int, dict[str, float]]] = []
    for label in ["all", *sorted(b for b in buckets if b != "all")]:
        qs = buckets[label]
        if not qs:
            continue
        names = per_query[qs[0]].keys()
        rows.append((label, len(qs), {n: statistics.fmean(per_query[q][n] for q in qs) for n in names}))
    return rows


def format_table(rows: Sequence[tuple[str, int, Mapping[str, float]]]) -> str:
    """Bảng văn bản căn cột để log/dán vào báo cáo."""
    if not rows:
        return "(không có câu hỏi nào)"
    names = list(rows[0][2])
    header = f"{'group':<18}{'n':>6}" + "".join(f"{n:>10}" for n in names)
    lines = [header, "-" * len(header)]
    for label, n, vals in rows:
        lines.append(f"{label:<18}{n:>6}" + "".join(f"{vals[m]:>10.4f}" for m in names))
    return "\n".join(lines)
