"""Tokenizer DÙNG CHUNG cho cả lúc dựng chỉ mục và lúc truy vấn.

Không ai được viết bản tokenize riêng: lệch tokenizer giữa index và query là lỗi
im lặng, làm điểm số tụt mà không báo lỗi.

Quy trình: lowercase -> tách từ theo ``\\w+`` -> bỏ stopword NLTK -> Porter stemmer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from functools import lru_cache, partial

from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

Analyzer = Callable[[str], list[str]]

_TOKEN_RE = re.compile(r"\w+")
_STEMMER = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    """Danh sách stopword tiếng Anh của NLTK (tự tải nếu chưa có)."""
    from nltk.corpus import stopwords as nltk_stopwords

    try:
        words = nltk_stopwords.words("english")
    except LookupError:
        import nltk

        logger.warning("Chưa có NLTK stopwords, đang tải về...")
        nltk.download("stopwords", quiet=True)
        words = nltk_stopwords.words("english")
    return frozenset(words)


@lru_cache(maxsize=1_000_000)
def _stem(token: str) -> str:
    """Porter stem có cache (stemmer của NLTK chậm, token lặp lại rất nhiều)."""
    return _STEMMER.stem(token)


def tokenize(text: str, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Tách ``text`` thành danh sách term.

    ``stem=False`` để so sánh có/không stemming; ``remove_stopwords=False`` để giữ stopword.
    Stopword được lọc TRƯỚC khi stem (danh sách NLTK ở dạng chưa stem).
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        sw = stopwords()
        tokens = [t for t in tokens if t not in sw]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens


def make_analyzer(stem: bool = True, remove_stopwords: bool = True) -> Analyzer:
    """Trả hàm ``text -> terms`` đã cố định tham số tokenize."""
    return partial(tokenize, stem=stem, remove_stopwords=remove_stopwords)
