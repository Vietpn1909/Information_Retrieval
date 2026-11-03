"""Nén postings list.

``InvertedIndex`` luôn lưu postings qua một ``PostingsCodec``. Hiện dùng
``IdentityCodec`` (không nén). [M1] cài 4 hàm delta/VByte bên dưới rồi đổi
``codec: delta_vbyte`` trong config — không cần sửa ``inverted.py``.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import Any, Protocol

IntSeq = Sequence[int]
Decoded = tuple[IntSeq, IntSeq, IntSeq]  # (doc_ids, tfs, positions phẳng)


class PostingsCodec(Protocol):
    """Mã hoá/giải mã postings của một term."""

    name: str

    def encode(self, doc_ids: IntSeq, tfs: IntSeq, positions: IntSeq) -> Any:
        """Mã hoá; ``positions`` là vị trí phẳng, mỗi doc chiếm ``tf`` phần tử liên tiếp."""
        ...

    def decode(self, blob: Any, with_positions: bool = False) -> Decoded:
        """Giải mã; ``with_positions=False`` cho phép trả positions rỗng để nhanh hơn."""
        ...


class IdentityCodec:
    """Không nén: giữ nguyên ba mảng ``array('I')``."""

    name = "identity"

    def encode(self, doc_ids: IntSeq, tfs: IntSeq, positions: IntSeq) -> Any:
        return (array("I", doc_ids), array("I", tfs), array("I", positions))

    def decode(self, blob: Any, with_positions: bool = False) -> Decoded:
        doc_ids, tfs, positions = blob
        return doc_ids, tfs, positions if with_positions else array("I")


# ---------------------------------------------------------------- [M1] TODO
def delta_encode(values: IntSeq) -> list[int]:
    """[M1] Dãy tăng -> khoảng cách (gap): [3, 7, 12] -> [3, 4, 5]."""
    raise NotImplementedError("M1: cài delta encoding")


def delta_decode(gaps: IntSeq) -> list[int]:
    """[M1] Khoảng cách -> dãy gốc (cộng dồn)."""
    raise NotImplementedError("M1: cài delta decoding")


def vbyte_encode(values: IntSeq) -> bytes:
    """[M1] Variable-byte: 7 bit dữ liệu/byte, bit cao đánh dấu byte cuối."""
    raise NotImplementedError("M1: cài VByte encoding")


def vbyte_decode(data: bytes) -> list[int]:
    """[M1] Giải mã VByte."""
    raise NotImplementedError("M1: cài VByte decoding")


class DeltaVByteCodec:
    """Delta cho doc_ids + VByte cho cả ba mảng (chạy được khi M1 cài xong các hàm trên)."""

    name = "delta_vbyte"

    def encode(self, doc_ids: IntSeq, tfs: IntSeq, positions: IntSeq) -> Any:
        return (vbyte_encode(delta_encode(doc_ids)), vbyte_encode(tfs), vbyte_encode(positions))

    def decode(self, blob: Any, with_positions: bool = False) -> Decoded:
        d, t, p = blob
        return delta_decode(vbyte_decode(d)), vbyte_decode(t), vbyte_decode(p) if with_positions else []


_CODECS: dict[str, type[IdentityCodec] | type[DeltaVByteCodec]] = {
    IdentityCodec.name: IdentityCodec,
    DeltaVByteCodec.name: DeltaVByteCodec,
}


def get_codec(name: str) -> PostingsCodec:
    """Tạo codec theo tên trong config."""
    if name not in _CODECS:
        raise KeyError(f"Codec '{name}' không tồn tại. Có: {sorted(_CODECS)}")
    return _CODECS[name]()
