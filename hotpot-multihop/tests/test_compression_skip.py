"""Test nén + skip pointer. Các test đánh dấu xfail sẽ thành XPASS khi M1 cài xong."""

import pytest

from hmr.index.compression import (
    IdentityCodec,
    delta_decode,
    delta_encode,
    get_codec,
    vbyte_decode,
    vbyte_encode,
)
from hmr.index.skip import intersect, intersect_with_skips

m1_todo = pytest.mark.xfail(raises=NotImplementedError, reason="M1 chưa cài", strict=False)


def test_identity_codec_roundtrip() -> None:
    codec = get_codec("identity")
    assert isinstance(codec, IdentityCodec)
    blob = codec.encode([1, 5, 9], [2, 1, 1], [0, 3, 7, 2])
    d, t, p = codec.decode(blob, with_positions=True)
    assert (list(d), list(t), list(p)) == ([1, 5, 9], [2, 1, 1], [0, 3, 7, 2])
    assert list(codec.decode(blob)[2]) == []


def test_unknown_codec() -> None:
    with pytest.raises(KeyError):
        get_codec("zip")


def test_intersect_two_pointers() -> None:
    assert intersect([1, 3, 5, 7, 9], [2, 3, 4, 9, 10]) == [3, 9]
    assert intersect([], [1]) == []


@m1_todo
def test_delta_roundtrip() -> None:
    assert delta_encode([3, 7, 12]) == [3, 4, 5]
    assert delta_decode([3, 4, 5]) == [3, 7, 12]


@m1_todo
def test_vbyte_roundtrip() -> None:
    values = [0, 5, 127, 128, 16384, 2**31 - 1]
    assert vbyte_decode(vbyte_encode(values)) == values
    assert vbyte_encode([5]) == bytes([0x85])  # IIR hình 5.8: bit cao = byte cuối


@m1_todo
def test_intersect_with_skips_matches_plain() -> None:
    a, b = list(range(0, 1000, 3)), list(range(0, 1000, 7))
    assert intersect_with_skips(a, b, 18, 12) == intersect(a, b)
