"""Test truy xuất dày / lai / rerank [M3 – Khiêm]. Không tải model: dùng bộ mã hoá giả."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import yaml

from hmr.common.config import load_config
from hmr.index.inverted import InvertedIndex
from hmr.retrieval import dense, rerank
from hmr.retrieval.dense import DenseRetriever, build_faiss, embedding_paths, file_md5, load_embeddings, passage_text
from hmr.retrieval.factory import build_retriever, build_system
from hmr.retrieval.hybrid import HybridRetriever, fuse, fusion_weights, linear_fuse, normalize, rrf_fuse
from hmr.retrieval.rerank import RerankRetriever, rerank_ranking

VOCAB = ["apple", "banana", "cherry", "durian"]


def bow_encode(texts: Sequence[str]) -> np.ndarray:
    """Bộ mã hoá giả: túi từ trên VOCAB, chuẩn hoá L2 (cosine = độ trùng từ)."""
    m = np.array([[t.lower().split().count(w) for w in VOCAB] for t in texts], dtype=np.float32)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    return m


DOCS = ["apple apple", "banana", "cherry banana", "durian cherry", "apple banana"]


# ------------------------------------------------------------------ dense
@pytest.mark.parametrize("spec", [{"kind": "flat"}, {"kind": "hnsw", "M": 8, "ef_search": 64}])
def test_dense_search_returns_doc_ids_by_cosine(spec: dict) -> None:
    retr = DenseRetriever(bow_encode, build_faiss(bow_encode(DOCS), spec))
    res = retr.search("apple", 3)
    assert res[0][0] == 0 and res[0][1] == pytest.approx(1.0, abs=1e-5)
    assert res[1][0] == 4  # "apple banana": cosine 1/√2
    assert [s for _, s in res] == sorted((s for _, s in res), reverse=True)
    # search_batch trùng search từng câu
    assert retr.search_batch(["apple", "durian"], 2) == [retr.search("apple", 2), retr.search("durian", 2)]


def test_dense_k_larger_than_corpus_and_ivf() -> None:
    emb = bow_encode(DOCS)
    retr = DenseRetriever(bow_encode, build_faiss(emb, {"kind": "ivf", "nlist": 2, "nprobe": 2}))
    res = retr.search("banana", 50)
    assert len(res) <= len(DOCS) and all(0 <= d < len(DOCS) for d, _ in res)
    with pytest.raises(ValueError):
        build_faiss(emb, {"kind": "lsh"})


def test_passage_text_format() -> None:
    doc = {"title": "Shirley Temple", "text": "An actress."}
    assert passage_text(doc, "{title}. {text}") == "Shirley Temple. An actress."
    assert passage_text(doc, "{text}", "passage: ") == "passage: An actress."


# ------------------------------------------------------------------ hybrid
def test_rrf_by_hand() -> None:
    a = [(1, 9.0), (2, 5.0)]
    b = [(2, 0.9), (3, 0.8)]
    out = dict(rrf_fuse([a, b], rrf_k=60, k=10))
    assert out[2] == pytest.approx(1 / 62 + 1 / 61)
    assert out[1] == pytest.approx(1 / 61) and out[3] == pytest.approx(1 / 62)
    assert [d for d, _ in rrf_fuse([a, b], 60, k=10)] == [2, 1, 3]
    # trọng số
    assert [d for d, _ in rrf_fuse([a, b], 60, [0.0, 1.0], k=1)] == [2]
    with pytest.raises(ValueError):
        rrf_fuse([a, b], 60, [1.0])


def test_normalize_and_linear() -> None:
    norm, floor = normalize([(1, 10.0), (2, 5.0), (3, 0.0)], "minmax")
    assert norm == {1: 1.0, 2: 0.5, 3: 0.0} and floor == 0.0
    z, zfloor = normalize([(1, 2.0), (2, 0.0)], "zscore")
    assert z == {1: 1.0, 2: -1.0} and zfloor == -1.0
    assert normalize([], "minmax") == ({}, 0.0)
    a = [(1, 10.0), (2, 0.0)]
    b = [(2, 1.0), (3, 0.0)]
    # alpha = 1 -> chỉ a; alpha = 0 -> chỉ b
    assert linear_fuse([a, b], [1.0, 0.0], k=1)[0][0] == 1
    assert linear_fuse([a, b], [0.0, 1.0], k=1)[0][0] == 2
    assert dict(linear_fuse([a, b], [0.5, 0.5], k=3))[2] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        normalize(a, "softmax")


def test_fusion_weights_and_fuse() -> None:
    assert fusion_weights({"fusion": "linear", "alpha": 0.3}, 2) == pytest.approx([0.3, 0.7])
    assert fusion_weights({"fusion": "rrf", "weights": [1, 2]}, 2) == [1.0, 2.0]
    assert fusion_weights({"fusion": "rrf"}, 3) == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        fusion_weights({"fusion": "linear", "alpha": 0.3}, 3)
    with pytest.raises(ValueError):
        fuse([[(1, 1.0)]], {"fusion": "max"}, 5)


class ListRetriever:
    def __init__(self, ranking: list[tuple[int, float]]) -> None:
        self.ranking = ranking
        self.calls: list[int] = []

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        self.calls.append(k)
        return self.ranking[:k]


def test_hybrid_retriever_uses_depth() -> None:
    a, b = ListRetriever([(1, 3.0), (2, 2.0), (3, 1.0)]), ListRetriever([(3, 0.9), (4, 0.1)])
    h = HybridRetriever([a, b], {"fusion": "rrf", "rrf_k": 1}, depth=50)
    res = h.search("q", 2)
    assert a.calls == [50] and len(res) == 2
    assert res[0][0] == 3  # 1/(1+3) + 1/(1+1) > 1/(1+1)


# ------------------------------------------------------------------ rerank
def test_rerank_head_reordered_tail_kept() -> None:
    texts = ["x", "yy", "zzz", "wwww", "v"]
    scorer = lambda pairs: [float(len(p[1])) for p in pairs]  # noqa: E731 — đoạn dài điểm cao
    cand = [(0, 9.0), (1, 8.0), (2, 7.0), (4, 6.0), (3, 5.0)]
    out = rerank_ranking("q", cand, texts, scorer, depth=3, k=5)
    assert [d for d, _ in out] == [2, 1, 0, 4, 3]
    scores = [s for _, s in out]
    assert all(x > y for x, y in zip(scores, scores[1:]))  # giảm ngặt
    assert len(rerank_ranking("q", cand, texts, scorer, depth=3, k=2)) == 2
    retr = RerankRetriever(ListRetriever(cand), scorer, texts, depth=2)
    assert [d for d, _ in retr.search("q", 3)] == [1, 0, 2]


# ------------------------------------------------------------------ factory + file vectơ
@pytest.fixture
def m3_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_index: InvertedIndex) -> dict:
    """Cấu hình đồ chơi: corpus 5 đoạn, vectơ túi từ, model giả (không tải gì)."""
    (tmp_path / "config").mkdir()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"doc_id": i, "title": f"T{i}", "text": t}) + "\n" for i, t in enumerate(DOCS)))
    idx = InvertedIndex(stem=False, remove_stopwords=False, include_title=False)
    for i, t in enumerate(DOCS):
        idx.add_document(i, t.split())
    idx.finalize()
    idx.save(tmp_path / "idx" / "toy.pkl")
    raw = {
        "paths": {"corpus": "corpus.jsonl", "dense_dir": "dense", "index_dir": "idx"},
        "retrievers": {
            "bm25": {"type": "bm25", "index": "toy", "k1": 0.9, "b": 1.0},
            "d": {"type": "dense", "model": "fake", "faiss": {"kind": "flat"}},
            "h": {"type": "hybrid", "components": ["bm25", "d"], "fusion": "rrf", "rrf_k": 60, "depth": 10},
            "r": {"type": "rerank", "base": "h", "model": "ce", "rerank_depth": 3},
        },
        "dense": {
            "models": {"fake": {"name": "fake/model", "max_length": 32}},
            "rerankers": {"ce": {"name": "fake/ce", "max_length": 32}},
            "passage_format": "{text}",
            "batch_size": 8,
        },
    }
    path = tmp_path / "config" / "m3.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)

    class FakeEncoder:
        def __init__(self, *args: object) -> None: ...

        def __call__(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
            return bow_encode(texts)

    monkeypatch.setattr(dense, "SentenceEncoder", FakeEncoder)
    monkeypatch.setattr(rerank, "CrossEncoderScorer", lambda *a: (lambda pairs: [-float(len(p[1])) for p in pairs]))
    return cfg


def test_encode_load_and_factory(m3_cfg: dict) -> None:
    with pytest.raises(FileNotFoundError, match="build_dense.py"):
        build_retriever(m3_cfg, "d")
    docs = [{"doc_id": i, "title": f"T{i}", "text": t} for i, t in enumerate(DOCS)]
    emb = dense.encode_corpus(m3_cfg, "fake", docs)
    npy, meta = embedding_paths(m3_cfg, "fake")
    assert npy.exists() and json.loads(meta.read_text())["corpus_md5"] == file_md5(m3_cfg["paths"]["corpus"])
    assert np.allclose(load_embeddings(m3_cfg, "fake"), emb)

    d = build_retriever(m3_cfg, "d")
    assert d.search("cherry", 1)[0][0] in (2, 3)
    h = build_retriever(m3_cfg, "h")
    assert {doc for doc, _ in h.search("apple", 5)} >= {0, 4}
    r = build_retriever(m3_cfg, "r")
    out = r.search("apple banana", 5)
    assert len(out) >= 2
    # factory bọc adapter để ra RetrievalResult như mọi cấu hình đơn bước
    res = build_system(m3_cfg, "h", [f"T{i}" for i in range(len(DOCS))]).search("apple", 3)
    assert res.ranked_passages[0].rank == 1 and len(res.hops) == 1


def test_embeddings_rejected_when_corpus_changed(m3_cfg: dict) -> None:
    docs = [{"doc_id": i, "title": f"T{i}", "text": t} for i, t in enumerate(DOCS)]
    dense.encode_corpus(m3_cfg, "fake", docs)
    Path(m3_cfg["paths"]["corpus"]).write_text("{}\n")
    with pytest.raises(ValueError, match="corpus khác"):
        load_embeddings(m3_cfg, "fake")


def test_encode_rejects_non_contiguous_doc_ids(m3_cfg: dict) -> None:
    with pytest.raises(ValueError, match="liên tục"):
        dense.encode_corpus(m3_cfg, "fake", [{"doc_id": 1, "title": "a", "text": "b"}])


# ------------------------------------------------------------------ dữ liệu thật
@pytest.mark.data
def test_real_embeddings_aligned_with_corpus() -> None:
    """Vectơ thật (nếu đã dựng) có đúng số dòng = số đoạn, đã chuẩn hoá, gắn đúng corpus."""
    cfg = load_config()
    found = False
    for key in cfg["dense"]["models"]:
        npy, meta = embedding_paths(cfg, key)
        if not npy.exists():
            continue
        found = True
        m = json.loads(meta.read_text())
        emb = np.load(npy, mmap_mode="r")
        assert emb.shape == (m["n_docs"], m["dim"]) and m["n_docs"] == 105862
        assert np.allclose(np.linalg.norm(emb[:1000], axis=1), 1.0, atol=1e-3)
        assert m["corpus_md5"] == cfg["data"]["checksums"]["corpus"]
    if not found:
        pytest.skip("Chưa chạy build_dense.py")
