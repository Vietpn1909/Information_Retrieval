"""Gióng hàng ``supporting_facts`` (title, sent_id) sang ``doc_id`` của corpus.

Đây là chỗ hay sai nhất của loại project này nên kiểm tra rất chặt:
- title phải có trong corpus;
- câu tại (doc_id, sent_id) trong corpus phải TRÙNG câu gốc của chính câu hỏi
  (so sau khi chuẩn hoá khoảng trắng).
Riêng sent_id vượt quá số câu ngay trong context gốc là nhiễu nhãn của HotpotQA
(có ~1/2500 câu hỏi), không phải lỗi pipeline: fact đó bị loại và ghi vào báo cáo.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from hmr.data.corpus import Corpus, normalize_ws

logger = logging.getLogger(__name__)

OnError = Literal["error", "drop"]


class AlignmentError(RuntimeError):
    """Có supporting fact không gióng hàng được sang corpus."""


@dataclass(frozen=True)
class AlignmentIssue:
    qid: str
    title: str
    sent_id: int
    reason: str


@dataclass
class AlignmentReport:
    n_questions: int = 0
    n_facts: int = 0
    n_aligned: int = 0
    failures: list[AlignmentIssue] = field(default_factory=list)  # lỗi pipeline
    source_noise: list[AlignmentIssue] = field(default_factory=list)  # nhiễu nhãn gốc

    @property
    def rate(self) -> float:
        """Tỉ lệ fact gióng hàng được (không tính nhiễu nhãn gốc)."""
        denom = self.n_facts - len(self.source_noise)
        return self.n_aligned / denom if denom else 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rate"] = self.rate
        return d


def align_example(
    example: Mapping[str, Any], corpus: Corpus
) -> tuple[list[dict[str, Any]], list[AlignmentIssue], list[AlignmentIssue]]:
    """Gióng hàng một câu hỏi.

    Trả ``(facts, failures, source_noise)``; mỗi fact là ``{title, sent_id, doc_id}``.
    """
    qid = example["id"]
    own_ctx = dict(zip(example["context"]["title"], example["context"]["sentences"], strict=True))
    sf = example["supporting_facts"]
    facts: list[dict[str, Any]] = []
    failures: list[AlignmentIssue] = []
    noise: list[AlignmentIssue] = []
    for title, sent_id in zip(sf["title"], sf["sent_id"], strict=True):
        own = own_ctx.get(title)
        if own is not None and sent_id >= len(own):
            noise.append(AlignmentIssue(qid, title, sent_id, f"sent_id >= {len(own)} câu trong context gốc"))
            continue
        doc_id = corpus.title_to_id.get(title)
        if doc_id is None:
            failures.append(AlignmentIssue(qid, title, sent_id, "title không có trong corpus"))
            continue
        sents = corpus.docs[doc_id]["sentences"]
        if sent_id >= len(sents):
            failures.append(AlignmentIssue(qid, title, sent_id, f"sent_id >= {len(sents)} câu trong corpus"))
            continue
        if own is not None and normalize_ws(sents[sent_id]) != normalize_ws(own[sent_id]):
            failures.append(AlignmentIssue(qid, title, sent_id, "câu trong corpus khác câu gốc"))
            continue
        facts.append({"title": title, "sent_id": sent_id, "doc_id": doc_id})
    return facts, failures, noise


def align_examples(
    examples: Iterable[tuple[str, Mapping[str, Any]]],
    corpus: Corpus,
    on_error: OnError = "error",
) -> tuple[list[dict[str, Any]], AlignmentReport]:
    """Gióng hàng mọi câu hỏi, trả bản ghi ``questions.jsonl`` và báo cáo.

    ``on_error="error"``: ném ``AlignmentError`` nếu có bất kỳ lỗi nào (fail sớm).
    ``on_error="drop"``: bỏ câu hỏi có lỗi và log cảnh báo.
    """
    report = AlignmentReport()
    records: list[dict[str, Any]] = []
    for split, ex in examples:
        report.n_questions += 1
        report.n_facts += len(ex["supporting_facts"]["title"])
        facts, failures, noise = align_example(ex, corpus)
        report.n_aligned += len(facts)
        report.failures.extend(failures)
        report.source_noise.extend(noise)
        if failures:
            continue
        gold_doc_ids = list(dict.fromkeys(f["doc_id"] for f in facts))
        records.append(
            {
                "qid": ex["id"],
                "split": split,
                "question": ex["question"],
                "answer": ex["answer"],
                "type": ex["type"],
                "level": ex["level"],
                "gold_doc_ids": gold_doc_ids,
                "supporting_facts": facts,
            }
        )
    for issue in report.source_noise:
        logger.warning("Nhiễu nhãn gốc, bỏ fact: %s", issue)
    if report.failures:
        for issue in report.failures[:20]:
            logger.error("Gióng hàng lỗi: %s", issue)
        msg = f"{len(report.failures)} supporting fact không gióng hàng được (rate={report.rate:.4%})"
        if on_error == "error":
            raise AlignmentError(msg)
        logger.warning("%s — đã bỏ %d câu hỏi", msg, report.n_questions - len(records))
    return records, report
