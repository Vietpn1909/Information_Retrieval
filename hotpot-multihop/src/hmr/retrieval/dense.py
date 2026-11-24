"""Truy xuất dày [M3 – Khiêm]: bi-encoder (BGE-small) + FAISS.

- Mỗi đoạn văn được mã hoá một lần (``scripts/build_dense.py``) thành vectơ đã chuẩn hoá L2,
  lưu ``data/dense/<model>.npy`` — dòng i của ma trận **là** ``doc_id = i`` (cùng thứ tự với
  ``corpus.jsonl``), nên FAISS trả về chỉ số dòng = ``doc_id``, không cần bảng ánh xạ.
- Điểm = tích vô hướng giữa vectơ truy vấn và vectơ đoạn = cosine (vì đã chuẩn hoá).
- Chỉ mục FAISS dựng lúc nạp từ ma trận vectơ: ``flat`` (chính xác), ``hnsw``, ``ivf`` (xấp xỉ).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hmr.common.config import Config

logger = logging.getLogger(__name__)

# Hàm mã hoá: danh sách chuỗi -> ma trận float32 (n, dim) đã chuẩn hoá L2
EncodeFn = Callable[[Sequence[str]], np.ndarray]
Ranking = list[tuple[int, float]]


# ------------------------------------------------------------------ cấu hình / đường dẫn
def model_config(cfg: Config, key: str) -> dict[str, Any]:
    """Cấu hình bi-encoder theo khoá trong ``dense.models``."""
    models = cfg["dense"]["models"]
    if key not in models:
        raise KeyError(f"Không có model '{key}' trong dense.models. Có: {sorted(models)}")
    return models[key]


def embedding_paths(cfg: Config, key: str) -> tuple[Path, Path]:
    """``(ma trận .npy, metadata .json)`` của model ``key``."""
    base = Path(cfg["paths"]["dense_dir"]) / key
    return base.with_suffix(".npy"), base.with_suffix(".json")


def file_md5(path: str | Path) -> str:
    """md5 của file (dùng để gắn vectơ với đúng phiên bản corpus)."""
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def passage_text(doc: Mapping[str, Any], fmt: str, prefix: str = "") -> str:
    """Văn bản đưa vào bi-encoder cho một đoạn (mặc định ``"{title}. {text}"``)."""
    return prefix + fmt.format(title=doc["title"], text=doc["text"])


def configure_threads(cfg: Config) -> None:
    """Đặt số luồng torch/faiss nếu cấu hình ``dense.threads`` khác null."""
    n = cfg["dense"].get("threads")
    if not n:
        return
    import faiss
    import torch

    torch.set_num_threads(int(n))
    faiss.omp_set_num_threads(int(n))


# ------------------------------------------------------------------ bộ mã hoá
class SentenceEncoder:
    """Bọc ``SentenceTransformer``: thêm prefix, chuẩn hoá L2, trả float32."""

    def __init__(self, name: str, prefix: str, batch_size: int, max_length: int, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device)
        self.model.max_seq_length = max_length
        self.prefix = prefix
        self.batch_size = batch_size

    def __call__(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        emb = self.model.encode(
            [self.prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)


def make_encoder(cfg: Config, key: str, role: str) -> SentenceEncoder:
    """Bộ mã hoá cho ``role`` = ``"query"`` hoặc ``"passage"`` (khác nhau ở prefix)."""
    mcfg = model_config(cfg, key)
    dcfg = cfg["dense"]
    return SentenceEncoder(
        mcfg["name"],
        mcfg.get(f"{role}_prefix", "") or "",
        int(dcfg["batch_size"]),
        int(mcfg["max_length"]),
        dcfg.get("device", "cpu"),
    )


# ------------------------------------------------------------------ vectơ đoạn văn
def encode_corpus(cfg: Config, key: str, docs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Mã hoá toàn bộ kho rồi ghi ``.npy`` + ``.json``; kiểm tra ``doc_id`` = số thứ tự dòng."""
    for i, d in enumerate(docs):
        if d["doc_id"] != i:
            raise ValueError(f"doc_id phải liên tục 0..N-1 theo thứ tự dòng; dòng {i} có doc_id {d['doc_id']}")
    encoder = make_encoder(cfg, key, "passage")
    fmt = cfg["dense"]["passage_format"]
    emb = encoder([passage_text(d, fmt) for d in docs], show_progress=True)
    npy, meta = embedding_paths(cfg, key)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, emb)
    meta.write_text(
        json.dumps(
            {
                "model": model_config(cfg, key)["name"],
                "n_docs": int(emb.shape[0]),
                "dim": int(emb.shape[1]),
                "passage_format": fmt,
                "corpus_md5": file_md5(cfg["paths"]["corpus"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Đã ghi %s %s", npy, emb.shape)
    return emb


def load_embeddings(cfg: Config, key: str, check_corpus: bool = True) -> np.ndarray:
    """Nạp ma trận vectơ đoạn; báo lỗi nếu chưa dựng hoặc dựng từ corpus khác."""
    npy, meta_path = embedding_paths(cfg, key)
    if not npy.exists():
        raise FileNotFoundError(f"Chưa có vectơ {npy}; chạy: uv run python scripts/build_dense.py --model {key}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = model_config(cfg, key)["name"]
    if meta["model"] != expected:
        raise ValueError(f"{npy} được mã hoá bằng {meta['model']}, cấu hình yêu cầu {expected} — dựng lại")
    if check_corpus and meta.get("corpus_md5") != file_md5(cfg["paths"]["corpus"]):
        raise ValueError(f"{npy} được dựng từ corpus khác (md5 lệch) — chạy lại build_dense.py")
    emb = np.load(npy)
    logger.info("Nạp vectơ %s %s", npy, emb.shape)
    return np.ascontiguousarray(emb, dtype=np.float32)


# ------------------------------------------------------------------ FAISS
def build_faiss(emb: np.ndarray, spec: Mapping[str, Any]) -> Any:
    """Dựng chỉ mục FAISS inner-product theo ``spec['kind']`` ∈ {flat, hnsw, ivf}."""
    import faiss

    dim = emb.shape[1]
    kind = spec.get("kind", "flat")
    if kind == "flat":
        index = faiss.IndexFlatIP(dim)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, int(spec.get("M", 32)), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(spec.get("ef_construction", 200))
        index.hnsw.efSearch = int(spec.get("ef_search", 128))
    elif kind == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, int(spec.get("nlist", 1024)), faiss.METRIC_INNER_PRODUCT)
        index.train(emb)
        index.nprobe = int(spec.get("nprobe", 32))
    else:
        raise ValueError(f"Kiểu FAISS chưa hỗ trợ: {kind} (flat | hnsw | ivf)")
    index.add(emb)
    return index


def set_search_param(index: Any, kind: str, value: int) -> None:
    """Đổi tham số lúc tìm (efSearch của HNSW / nprobe của IVF) mà không dựng lại."""
    if kind == "hnsw":
        index.hnsw.efSearch = int(value)
    elif kind == "ivf":
        index.nprobe = int(value)


# ------------------------------------------------------------------ bộ truy xuất
class DenseRetriever:
    """Cài đặt ``Retriever``: mã hoá truy vấn → tìm k láng giềng gần nhất trong FAISS."""

    def __init__(self, encode_query: EncodeFn, index: Any) -> None:
        self.encode_query = encode_query
        self.index = index

    def search(self, query: str, k: int) -> Ranking:
        """Top-k ``(doc_id, cosine)`` giảm dần."""
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int) -> list[Ranking]:
        """Như ``search`` cho nhiều truy vấn cùng lúc (mã hoá theo lô — nhanh hơn nhiều)."""
        if not queries:
            return []
        return self.search_vectors(self.encode_query(list(queries)), k)

    def search_vectors(self, qvecs: np.ndarray, k: int) -> list[Ranking]:
        """Tìm với vectơ truy vấn đã mã hoá sẵn (dùng khi đo riêng thời gian tìm ANN)."""
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(np.ascontiguousarray(qvecs, dtype=np.float32), k)
        # FAISS trả -1 khi không đủ k kết quả (IVF với nprobe nhỏ) — bỏ đi
        return [
            [(int(d), float(s)) for d, s in zip(row_ids, row_scores) if d >= 0]
            for row_ids, row_scores in zip(ids, scores)
        ]


def build(cfg: Config, rcfg: Mapping[str, Any]) -> DenseRetriever:
    """Builder cho ``type: dense`` (được ``factory.build_retriever`` gọi)."""
    configure_threads(cfg)
    key = rcfg["model"]
    emb = load_embeddings(cfg, key)
    index = build_faiss(emb, rcfg.get("faiss", {"kind": "flat"}))
    return DenseRetriever(make_encoder(cfg, key, "query"), index)
