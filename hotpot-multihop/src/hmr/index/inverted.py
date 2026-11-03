"""Chỉ mục ngược có vị trí từ, tự cài đặt.

Cấu trúc mỗi term: ``(doc_ids, tfs, positions)`` — doc_ids tăng dần; positions là
mảng phẳng, doc thứ i chiếm ``tfs[i]`` phần tử liên tiếp. Ba mảng được lưu qua
``PostingsCodec`` (xem ``compression.py``). Vị trí tính trên dãy token SAU tokenize.
"""

from __future__ import annotations

import bisect
import logging
import pickle
from array import array
from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import accumulate
from pathlib import Path
from typing import Any

from tqdm import tqdm

from hmr.common.tokenizer import tokenize
from hmr.index.compression import PostingsCodec, get_codec

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


class InvertedIndex:
    """Chỉ mục ngược; cài đặt ``hmr.common.interfaces.Index``.

    doc_id phải được thêm theo thứ tự 0, 1, 2, ... nên postings tự động tăng dần.
    """

    def __init__(
        self,
        stem: bool = True,
        remove_stopwords: bool = True,
        include_title: bool = True,
        codec: str = "identity",
    ) -> None:
        self.stem = stem
        self.remove_stopwords = remove_stopwords
        self.include_title = include_title
        self.format_version = FORMAT_VERSION
        self._codec: PostingsCodec = get_codec(codec)
        self._building: dict[str, tuple[list[int], list[int], list[int]]] | None = {}
        self._postings: dict[str, Any] = {}
        self._df: dict[str, int] = {}
        self._doc_lens = array("I")
        self._total_len = 0

    # ------------------------------------------------------------ dựng chỉ mục
    def analyze(self, text: str) -> list[str]:
        """Tokenize với đúng tham số đã dùng khi dựng chỉ mục (dùng cho truy vấn)."""
        return tokenize(text, stem=self.stem, remove_stopwords=self.remove_stopwords)

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        """Thêm một tài liệu đã tokenize; doc_id phải bằng số tài liệu hiện có."""
        if self._building is None:
            raise RuntimeError("Chỉ mục đã finalize, không thêm được tài liệu")
        if doc_id != len(self._doc_lens):
            raise ValueError(f"doc_id phải liên tục: mong đợi {len(self._doc_lens)}, nhận {doc_id}")
        local: dict[str, list[int]] = defaultdict(list)
        for pos, term in enumerate(tokens):
            local[term].append(pos)
        for term, pos_list in local.items():
            entry = self._building.get(term)
            if entry is None:
                entry = ([], [], [])
                self._building[term] = entry
            entry[0].append(doc_id)
            entry[1].append(len(pos_list))
            entry[2].extend(pos_list)
        self._doc_lens.append(len(tokens))
        self._total_len += len(tokens)

    def finalize(self) -> None:
        """Chốt chỉ mục: mã hoá postings qua codec, giải phóng bộ đệm dựng."""
        if self._building is None:
            return
        for term, (doc_ids, tfs, positions) in self._building.items():
            self._postings[term] = self._codec.encode(doc_ids, tfs, positions)
            self._df[term] = len(doc_ids)
        self._building = None

    @classmethod
    def from_corpus(
        cls,
        docs: Iterable[Mapping[str, Any]],
        *,
        stem: bool = True,
        remove_stopwords: bool = True,
        include_title: bool = True,
        codec: str = "identity",
        total: int | None = None,
    ) -> InvertedIndex:
        """Dựng chỉ mục từ các bản ghi corpus (``doc_id``, ``title``, ``text``)."""
        index = cls(stem=stem, remove_stopwords=remove_stopwords, include_title=include_title, codec=codec)
        for doc in tqdm(docs, total=total, desc="Chỉ mục", unit="doc"):
            text = f"{doc['title']} {doc['text']}" if include_title else doc["text"]
            index.add_document(doc["doc_id"], index.analyze(text))
        index.finalize()
        logger.info("Chỉ mục: %d tài liệu, %d term, avgdl=%.1f", index.n_docs(), len(index._df), index.avg_doc_len())
        return index

    # ------------------------------------------------------------ interface Index
    def _blob(self, term: str) -> Any | None:
        if self._building is not None:
            raise RuntimeError("Chưa finalize chỉ mục")
        return self._postings.get(term)

    def postings(self, term: str) -> list[tuple[int, int]]:
        """``[(doc_id, tf)]`` sắp tăng theo doc_id."""
        blob = self._blob(term)
        if blob is None:
            return []
        doc_ids, tfs, _ = self._codec.decode(blob)
        return list(zip(doc_ids, tfs))

    def doc_len(self, doc_id: int) -> int:
        return self._doc_lens[doc_id]

    def avg_doc_len(self) -> float:
        return self._total_len / len(self._doc_lens) if self._doc_lens else 0.0

    def n_docs(self) -> int:
        return len(self._doc_lens)

    def df(self, term: str) -> int:
        return self._df.get(term, 0)

    def vocabulary(self) -> Iterable[str]:
        return self._df.keys()

    # ------------------------------------------------------------ vị trí từ
    def positional_postings(self, term: str) -> list[tuple[int, list[int]]]:
        """``[(doc_id, [vị trí...])]`` — nền cho truy vấn cụm từ / lân cận."""
        blob = self._blob(term)
        if blob is None:
            return []
        doc_ids, tfs, positions = self._codec.decode(blob, with_positions=True)
        offsets = [0, *accumulate(tfs)]
        return [(d, list(positions[offsets[i] : offsets[i + 1]])) for i, d in enumerate(doc_ids)]

    def positions(self, term: str, doc_id: int) -> list[int]:
        """Vị trí của ``term`` trong ``doc_id`` (tìm nhị phân trên postings)."""
        blob = self._blob(term)
        if blob is None:
            return []
        doc_ids, tfs, positions = self._codec.decode(blob, with_positions=True)
        i = bisect.bisect_left(doc_ids, doc_id)
        if i == len(doc_ids) or doc_ids[i] != doc_id:
            return []
        start = sum(tfs[:i])
        return list(positions[start : start + tfs[i]])

    # ------------------------------------------------------------ lưu / nạp
    def save(self, path: str | Path) -> None:
        """Lưu bằng pickle (tự finalize nếu chưa)."""
        self.finalize()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> InvertedIndex:
        """Nạp từ pickle, kiểm tra phiên bản định dạng."""
        with Path(path).open("rb") as f:
            index = pickle.load(f)
        if not isinstance(index, cls) or index.format_version != FORMAT_VERSION:
            raise ValueError(f"{path} không phải InvertedIndex phiên bản {FORMAT_VERSION}; hãy dựng lại")
        return index
