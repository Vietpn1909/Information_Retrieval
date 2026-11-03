"""Dữ liệu đồ chơi theo đúng định dạng HotpotQA trên HuggingFace."""

from __future__ import annotations

from typing import Any

import pytest

from hmr.index.inverted import InvertedIndex


@pytest.fixture
def toy_examples() -> list[dict[str, Any]]:
    return [
        {
            "id": "q1",
            "question": "Which country is the director of Film Alpha from?",
            "answer": "France",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["Film Alpha", "Jean Dupont"], "sent_id": [0, 1]},
            "context": {
                "title": ["Film Alpha", "Jean Dupont", "Banana"],
                "sentences": [
                    ["Film Alpha is a 1999 drama directed by Jean Dupont.", " It won two awards."],
                    ["Jean Dupont is a film director.", " He was born in France."],
                    ["Banana is a fruit."],
                ],
            },
        },
        {
            "id": "q2",
            "question": "Are Jean Dupont and Maria Rossi both directors?",
            "answer": "yes",
            "type": "comparison",
            "level": "hard",
            "supporting_facts": {"title": ["Jean Dupont", "Maria Rossi"], "sent_id": [0, 0]},
            "context": {
                # Cùng title "Jean Dupont" nhưng khác khoảng trắng (NBSP) — như dữ liệu thật
                "title": ["Jean Dupont", "Maria Rossi"],
                "sentences": [
                    ["Jean\xa0Dupont is a film director.", " He was born in France."],
                    ["Maria Rossi is an Italian film director."],
                ],
            },
        },
    ]


@pytest.fixture
def toy_index() -> InvertedIndex:
    """3 tài liệu, không stem/không bỏ stopword để dễ tính tay."""
    index = InvertedIndex(stem=False, remove_stopwords=False, include_title=False)
    index.add_document(0, ["apple", "banana", "apple"])
    index.add_document(1, ["banana", "cherry"])
    index.add_document(2, ["cherry", "durian", "elder", "fig"])
    index.finalize()
    return index
