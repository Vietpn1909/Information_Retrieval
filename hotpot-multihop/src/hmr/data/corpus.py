"""Gộp context của các câu hỏi thành MỘT corpus chung, khử trùng lặp theo title."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hmr.common.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def normalize_ws(text: str) -> str:
    """Gộp mọi khoảng trắng (kể cả NBSP) thành một dấu cách."""
    return " ".join(text.split())


def passage_text(sentences: Sequence[str]) -> str:
    """Nối các câu thành văn bản đoạn."""
    return normalize_ws(" ".join(sentences))


@dataclass
class Corpus:
    """Corpus đoạn văn; ``doc_id`` = vị trí trong ``docs`` (0..N-1)."""

    docs: list[dict[str, Any]] = field(default_factory=list)
    title_to_id: dict[str, int] = field(default_factory=dict)
    n_conflicts: int = 0  # số lần gặp lại title với nội dung khác

    def __len__(self) -> int:
        return len(self.docs)

    def add_passage(self, title: str, sentences: Sequence[str]) -> int:
        """Thêm đoạn nếu title chưa có; trả doc_id.

        Nếu title đã có nhưng nội dung khác: giữ bản có nhiều câu hơn
        (trên HotpotQA thực tế các xung đột chỉ khác khoảng trắng, số câu bằng nhau).
        """
        sentences = list(sentences)
        doc_id = self.title_to_id.get(title)
        if doc_id is None:
            doc_id = len(self.docs)
            self.title_to_id[title] = doc_id
            self.docs.append(
                {"doc_id": doc_id, "title": title, "sentences": sentences, "text": passage_text(sentences)}
            )
            return doc_id
        old = self.docs[doc_id]
        if old["sentences"] != sentences:
            self.n_conflicts += 1
            if len(sentences) > len(old["sentences"]):
                old["sentences"] = sentences
                old["text"] = passage_text(sentences)
        return doc_id

    def add_example(self, example: Mapping[str, Any]) -> None:
        """Thêm toàn bộ context (10 đoạn) của một câu hỏi HotpotQA."""
        ctx = example["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"], strict=True):
            self.add_passage(title, sentences)

    @classmethod
    def from_examples(cls, examples: Iterable[Mapping[str, Any]]) -> Corpus:
        """Dựng corpus từ danh sách câu hỏi."""
        corpus = cls()
        for ex in examples:
            corpus.add_example(ex)
        return corpus

    def save(self, path: str | Path) -> None:
        """Ghi ``corpus.jsonl``."""
        write_jsonl(path, self.docs)

    @classmethod
    def load(cls, path: str | Path) -> Corpus:
        """Nạp ``corpus.jsonl``; kiểm tra doc_id liên tục."""
        corpus = cls()
        for i, doc in enumerate(read_jsonl(path)):
            if doc["doc_id"] != i:
                raise ValueError(f"doc_id không liên tục tại dòng {i}: {doc['doc_id']}")
            corpus.docs.append(doc)
            corpus.title_to_id[doc["title"]] = i
        return corpus
