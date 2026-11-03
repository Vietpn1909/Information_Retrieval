"""Kiểm tra TRÊN DỮ LIỆU THẬT: 100% supporting_facts gióng hàng sang doc_id.

Tự skip nếu chưa chạy ``scripts/build_corpus.py``.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from hmr.common.config import load_config
from hmr.common.io import read_jsonl, read_qrels

CFG = load_config()
PATHS = CFG["paths"]
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not Path(PATHS["questions"]).exists(), reason="Chưa chạy build_corpus.py"),
]


@pytest.fixture(scope="module")
def corpus() -> list[dict[str, Any]]:
    return list(read_jsonl(PATHS["corpus"]))


@pytest.fixture(scope="module")
def questions() -> list[dict[str, Any]]:
    return list(read_jsonl(PATHS["questions"]))


@pytest.mark.parametrize("name", ["corpus", "questions", "qrels"])
def test_data_matches_reference_checksum(name: str) -> None:
    """Dữ liệu dựng lại phải trùng từng byte bản chuẩn (doc_id của mọi người khớp nhau)."""
    digest = hashlib.md5(Path(PATHS[name]).read_bytes()).hexdigest()
    assert digest == CFG["data"]["checksums"][name], (
        f"{PATHS[name]} khác bản chuẩn — kiểm tra data.hf_revision / seed / n_train, rồi chạy lại build_corpus.py"
    )


def test_every_supporting_fact_aligns(corpus: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    bad = []
    for q in questions:
        for f in q["supporting_facts"]:
            doc = corpus[f["doc_id"]]
            if doc["title"] != f["title"] or f["sent_id"] >= len(doc["sentences"]):
                bad.append((q["qid"], f))
    assert not bad, f"{len(bad)} fact gióng hàng sai, ví dụ: {bad[:5]}"


def test_report_is_100_percent() -> None:
    report = json.loads((Path(PATHS["data_dir"]) / "alignment_report.json").read_text(encoding="utf-8"))
    assert report["failures"] == []
    assert report["rate"] == 1.0
    assert report["n_aligned"] + len(report["source_noise"]) == report["n_facts"]


def test_titles_unique_and_ids_contiguous(corpus: list[dict[str, Any]]) -> None:
    assert [d["doc_id"] for d in corpus] == list(range(len(corpus)))
    assert len({d["title"] for d in corpus}) == len(corpus)


def test_two_gold_docs_and_qrels_consistent(questions: list[dict[str, Any]]) -> None:
    qrels = read_qrels(PATHS["qrels"])
    counts: dict[int, int] = defaultdict(int)
    for q in questions:
        counts[len(q["gold_doc_ids"])] += 1
        assert set(qrels[q["qid"]]) == {str(d) for d in q["gold_doc_ids"]}
    assert dict(counts) == {2: len(questions)}, "HotpotQA: mỗi câu có đúng 2 đoạn vàng"
    assert len(qrels) == len(questions)
