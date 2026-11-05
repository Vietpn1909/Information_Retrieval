"""Test nén postings [M1 – Nguyên]. Các test xfail sẽ thành XPASS khi M1 cài xong.

Test skip pointer nằm ở ``tests/test_skip.py`` [M2 – Việt].
"""

import pytest

from hmr.index.compression import (
    IdentityCodec,
    delta_decode,
    delta_encode,
    get_codec,
    vbyte_decode,
    vbyte_encode,
)

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


@m1_todo
def test_delta_roundtrip() -> None:
    assert delta_encode([3, 7, 12]) == [3, 4, 5]
    assert delta_decode([3, 4, 5]) == [3, 7, 12]


@m1_todo
def test_vbyte_roundtrip() -> None:
    values = [0, 5, 127, 128, 16384, 2**31 - 1]
    assert vbyte_decode(vbyte_encode(values)) == values
    assert vbyte_encode([5]) == bytes([0x85])  # IIR hình 5.8: bit cao = byte cuối
