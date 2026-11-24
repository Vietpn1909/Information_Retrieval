"""[M3 – Khiêm] Mã hoá toàn bộ kho đoạn văn bằng bi-encoder, ghi data/dense/<model>.npy.

Dòng i của ma trận là vectơ của doc_id = i. Chạy một lần sau build_corpus.py
(BGE-small trên CPU: ~15–30 phút cho 105 862 đoạn; GPU: ~1–2 phút).

Ví dụ: uv run python scripts/build_dense.py --model bge_small
"""

from __future__ import annotations

import argparse
import logging
import time

from hmr.common.config import load_config
from hmr.common.io import read_jsonl
from hmr.common.log import setup_logging
from hmr.retrieval.dense import configure_threads, embedding_paths, encode_corpus

logger = logging.getLogger("build_dense")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", default=["bge_small"], help="Khoá trong dense.models")
    parser.add_argument("--force", action="store_true", help="Mã hoá lại dù đã có file")
    parser.add_argument("--config-file", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    configure_threads(cfg)
    docs = list(read_jsonl(cfg["paths"]["corpus"]))
    for key in args.model:
        npy, _ = embedding_paths(cfg, key)
        if npy.exists() and not args.force:
            logger.info("%s đã có — bỏ qua (dùng --force để mã hoá lại)", npy)
            continue
        t0 = time.perf_counter()
        encode_corpus(cfg, key, docs)
        logger.info("%s: %d đoạn trong %.1f phút", key, len(docs), (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
