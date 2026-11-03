"""Bước 2: dựng chỉ mục ngược từ corpus.jsonl cho một hoặc nhiều biến thể trong ``indexes``."""

from __future__ import annotations

import argparse
import logging
import time

from hmr.common.config import index_path, load_config
from hmr.common.io import read_jsonl
from hmr.common.log import setup_logging
from hmr.index.inverted import InvertedIndex

logger = logging.getLogger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=None, help="YAML cấu hình (mặc định config/default.yaml)")
    parser.add_argument("--index", nargs="+", default=["stem"], help="Tên biến thể trong config.indexes")
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config_file)
    corpus_path = cfg["paths"]["corpus"]
    n_docs = sum(1 for _ in read_jsonl(corpus_path))
    for name in args.index:
        icfg = cfg["indexes"][name]
        t0 = time.perf_counter()
        index = InvertedIndex.from_corpus(read_jsonl(corpus_path), total=n_docs, **icfg)
        out = index_path(cfg, name)
        index.save(out)
        logger.info("Đã lưu chỉ mục '%s' -> %s (%.1fs)", name, out, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
