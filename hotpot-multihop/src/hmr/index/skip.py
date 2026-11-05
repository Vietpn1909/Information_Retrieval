"""Giao postings list và skip pointer [M2 – Việt] (IIR mục 2.3).

Skip pointer được đặt NGẦM ĐỊNH trên mảng doc_id: vị trí i là bội của ``step``
có con trỏ nhảy tới i + step (nếu còn trong mảng). Mặc định step = ⌊√P⌋.
Mọi hàm nhận ``stats`` để đếm phép so sánh doc_id và số lần nhảy — thước đo
không phụ thuộc máy cho thí nghiệm.

Ghi chú phối hợp [M1 – Nguyên]: skip ở đây chạy trên doc_id ĐÃ giải mã. Khi postings
nén VByte, skip pointer cần lưu (doc_id, byte offset) theo khối để nhảy mà không giải mã.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Sequence
from dataclasses import dataclass

from tqdm import tqdm

from hmr.common.interfaces import Index

StepSpec = int | str  # số nguyên hoặc "sqrt"


@dataclass(slots=True)
class IntersectStats:
    """Bộ đếm chi phí: phép so sánh doc_id và số lần nhảy theo skip pointer."""

    comparisons: int = 0
    skips: int = 0


def skip_step(n: int) -> int:
    """Bước nhảy heuristic ⌊√P⌋ cho postings dài P (tối thiểu 1)."""
    return max(1, math.isqrt(n))


def resolve_step(spec: StepSpec, n: int) -> int:
    """``"sqrt"`` -> ⌊√n⌋; số nguyên -> giữ nguyên (≤ 1 nghĩa là không có skip)."""
    if spec == "sqrt":
        return skip_step(n)
    if isinstance(spec, int):
        return max(1, spec)
    raise ValueError(f"Bước skip không hợp lệ: {spec!r}")


def intersect(a: Sequence[int], b: Sequence[int], stats: IntersectStats | None = None) -> list[int]:
    """Giao hai dãy doc_id tăng dần bằng hai con trỏ, O(|a| + |b|)."""
    i = j = comps = 0
    na, nb = len(a), len(b)
    out: list[int] = []
    while i < na and j < nb:
        x, y = a[i], b[j]
        comps += 1
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    if stats is not None:
        stats.comparisons += comps
    return out


def intersect_with_skips(
    a: Sequence[int],
    b: Sequence[int],
    step_a: int | None = None,
    step_b: int | None = None,
    stats: IntersectStats | None = None,
) -> list[int]:
    """Giao hai dãy tăng dần có skip pointer — thuật toán IIR hình 2.10.

    Con trỏ đang ở vị trí có skip và đích skip vẫn ≤ phần tử bên kia thì nhảy (lặp lại),
    ngược lại tiến một bước. Bước mặc định ⌊√P⌋ cho từng dãy.
    """
    na, nb = len(a), len(b)
    sa = step_a if step_a is not None else skip_step(na)
    sb = step_b if step_b is not None else skip_step(nb)
    i = j = comps = skips = 0
    out: list[int] = []
    while i < na and j < nb:
        x, y = a[i], b[j]
        comps += 1
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            jumped = False
            while sa > 1 and i % sa == 0 and i + sa < na:
                comps += 1
                if a[i + sa] > y:
                    break
                i += sa
                skips += 1
                jumped = True
            if not jumped:
                i += 1
        else:
            jumped = False
            while sb > 1 and j % sb == 0 and j + sb < nb:
                comps += 1
                if b[j + sb] > x:
                    break
                j += sb
                skips += 1
                jumped = True
            if not jumped:
                j += 1
    if stats is not None:
        stats.comparisons += comps
        stats.skips += skips
    return out


class SkipPostings:
    """Postings của một term dạng mảng (doc_ids, tfs) kèm skip pointer ngầm định."""

    __slots__ = ("doc_ids", "tfs", "step")

    def __init__(self, doc_ids: Sequence[int], tfs: Sequence[int], step: int | None = None) -> None:
        if len(doc_ids) != len(tfs):
            raise ValueError("doc_ids và tfs phải cùng độ dài")
        self.doc_ids = array("I", doc_ids)
        self.tfs = array("I", tfs)
        self.step = step if step is not None else skip_step(len(doc_ids))

    @classmethod
    def from_postings(cls, postings: Sequence[tuple[int, int]], step: StepSpec = "sqrt") -> SkipPostings:
        """Tạo từ ``Index.postings(term)``."""
        return cls([d for d, _ in postings], [tf for _, tf in postings], resolve_step(step, len(postings)))

    def __len__(self) -> int:
        return len(self.doc_ids)

    def lookup(self, candidates: Sequence[int], stats: IntersectStats | None = None) -> list[int]:
        """tf của từng ứng viên (đã sắp tăng) trong postings này; 0 nếu không có.

        Đây là phép giao "ứng viên ngắn × postings dài": chỉ postings dùng skip pointer.
        """
        ids, tfs, s = self.doc_ids, self.tfs, self.step
        n = len(ids)
        j = comps = skips = 0
        out: list[int] = []
        for c in candidates:
            while j < n:
                comps += 1
                if ids[j] >= c:
                    break
                if s > 1 and j % s == 0 and j + s < n:
                    comps += 1
                    if ids[j + s] <= c:
                        j += s
                        skips += 1
                        continue
                j += 1
            out.append(tfs[j] if j < n and ids[j] == c else 0)
        if stats is not None:
            stats.comparisons += comps
            stats.skips += skips
        return out


def build_skip_table(index: Index, step: StepSpec = "sqrt", show_progress: bool = True) -> dict[str, SkipPostings]:
    """Chuyển postings của MỌI term sang ``SkipPostings`` một lần (dùng khi chấm lại ứng viên)."""
    terms = list(index.vocabulary())
    return {
        t: SkipPostings.from_postings(index.postings(t), step)
        for t in tqdm(terms, desc="Skip pointer", unit="term", leave=False, disable=not show_progress)
    }
