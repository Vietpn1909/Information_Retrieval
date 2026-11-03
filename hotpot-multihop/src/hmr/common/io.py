"""Đọc/ghi JSONL, TREC run file và qrels.

TREC run: ``qid  Q0  doc_id  rank  score  run_name`` (rank bắt đầu từ 1).
Qrels:    ``qid  0  doc_id  rel``.
Khi đọc, doc_id được giữ dạng chuỗi vì pytrec_eval yêu cầu như vậy.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

Run = dict[str, dict[str, float]]
Qrels = dict[str, dict[str, int]]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Duyệt từng dòng JSON."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Ghi các bản ghi ra JSONL, trả số dòng đã ghi."""
    path = Path(path)
    _ensure_parent(path)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_trec_run(
    path: str | Path,
    rankings: Mapping[str, Sequence[tuple[int, float]]],
    run_name: str,
) -> None:
    """Ghi run file; mỗi ranking phải đã sắp giảm dần theo score."""
    path = Path(path)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for qid, ranking in rankings.items():
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def read_trec_run(path: str | Path) -> Run:
    """Đọc run file thành ``{qid: {doc_id: score}}``."""
    run: Run = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"Dòng run sai định dạng: {line!r}")
            qid, _, doc_id, _, score, _ = parts
            run.setdefault(qid, {})[doc_id] = float(score)
    return run


def write_qrels(path: str | Path, qrels: Mapping[str, Iterable[int]]) -> None:
    """Ghi qrels nhị phân: mỗi doc vàng một dòng ``qid 0 doc_id 1``."""
    path = Path(path)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for qid, doc_ids in qrels.items():
            for doc_id in doc_ids:
                f.write(f"{qid} 0 {doc_id} 1\n")


def read_qrels(path: str | Path) -> Qrels:
    """Đọc qrels thành ``{qid: {doc_id: rel}}``."""
    qrels: Qrels = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"Dòng qrels sai định dạng: {line!r}")
            qid, _, doc_id, rel = parts
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels
