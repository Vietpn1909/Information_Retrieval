"""Xác định đoạn văn hop 2 của câu hỏi bridge (HotpotQA không gán nhãn sẵn).

Heuristic, theo thứ tự ưu tiên:
1. ``title_link``: đoạn A là hop 2 nếu title của A được nhắc trong văn bản đoạn B
   (đoạn B chứa thực thể cầu nối) và điều ngược lại không đúng.
2. ``answer``: nếu chỉ một đoạn chứa câu trả lời thì đó là hop 2.
3. ``ambiguous``: không xác định được, trả None.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, NamedTuple

Hop2Method = Literal["title_link", "answer", "ambiguous"]

_PAREN_RE = re.compile(r"\s*\(.*\)\s*$")


class GoldPassage(NamedTuple):
    doc_id: int
    title: str
    text: str


def _core_title(title: str) -> str:
    """Bỏ phần phân biệt nghĩa: ``"Ed Wood (film)"`` -> ``"ed wood"``."""
    return _PAREN_RE.sub("", title).strip().lower()


def identify_hop2(gold: Sequence[GoldPassage], answer: str) -> tuple[int | None, Hop2Method]:
    """Trả ``(doc_id hop 2, phương pháp)``; chỉ xét câu có đúng 2 đoạn vàng."""
    if len(gold) != 2:
        return None, "ambiguous"
    a, b = gold
    a_in_b = bool(_core_title(a.title)) and _core_title(a.title) in b.text.lower()
    b_in_a = bool(_core_title(b.title)) and _core_title(b.title) in a.text.lower()
    if a_in_b != b_in_a:
        return (a.doc_id if a_in_b else b.doc_id), "title_link"
    ans = answer.strip().lower()
    if ans:
        a_has, b_has = ans in a.text.lower(), ans in b.text.lower()
        if a_has != b_has:
            return (a.doc_id if a_has else b.doc_id), "answer"
    return None, "ambiguous"
