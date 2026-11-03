"""Giao postings list: bản hai con trỏ cơ bản và bản có skip pointer [M1]."""

from __future__ import annotations

from collections.abc import Sequence


def intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Giao hai dãy doc_id tăng dần bằng hai con trỏ, O(|a| + |b|)."""
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


# ---------------------------------------------------------------- [M1] TODO
def skip_step(n: int) -> int:
    """[M1] Bước nhảy mặc định ~ sqrt(n) (theo IIR, mục 2.3)."""
    raise NotImplementedError("M1: chọn bước skip")


def intersect_with_skips(a: Sequence[int], b: Sequence[int], step_a: int, step_b: int) -> list[int]:
    """[M1] Giao hai dãy, nhảy cóc ``step`` phần tử khi doc_id ở đích skip vẫn nhỏ hơn."""
    raise NotImplementedError("M1: cài giao có skip pointer")
