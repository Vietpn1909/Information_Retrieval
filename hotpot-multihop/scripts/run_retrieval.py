"""Bước 3: chạy một cấu hình truy xuất trên một split, xuất TREC run file.

Ví dụ: python scripts/run_retrieval.py --config bm25 --split dev --k 10
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import statistics
from pathlib import Path

from tqdm import tqdm

from hmr.common.config import load_config
from hmr.common.io import read_jsonl, write_trec_run
from hmr.common.log import setup_logging
from hmr.retrieval.factory import build_system

logger = logging.getLogger("run_retrieval")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Tên cấu hình trong retrievers (bm25, tfidf, ...)")
    parser.add_argument("--split", choices=["dev", "train", "all"], default="dev")
    parser.add_argument("--k", type=int, default=None, help="Độ sâu run (mặc định retrieval.top_k)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu (thử nhanh)")
    parser.add_argument("--out", default=None, help="Mặc định runs/<config>.txt")
    parser.add_argument("--config-file", default=None, help="YAML cấu hình (mặc định config/default.yaml)")
    setup_logging()
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    paths = cfg["paths"]
    k = args.k or int(cfg["retrieval"]["top_k"])
    questions = [q for q in read_jsonl(paths["questions"]) if args.split == "all" or q["split"] == args.split]
    if args.limit:
        questions = questions[: args.limit]
    titles = [d["title"] for d in read_jsonl(paths["corpus"])]

    system = build_system(cfg, args.config, titles)
    results = {}
    for q in tqdm(questions, desc=f"{args.config}@{k}", unit="q"):
        results[q["qid"]] = dataclasses.replace(system.search(q["question"], k), qid=q["qid"])

    out = Path(args.out) if args.out else Path(paths["runs_dir"]) / f"{args.config}.txt"
    write_trec_run(out, {qid: r.as_ranking() for qid, r in results.items()}, run_name=args.config)
    lat = [r.latency_ms for r in results.values()]
    logger.info(
        "Đã ghi %d câu hỏi -> %s | latency trung bình %.1f ms, p95 %.1f ms",
        len(results),
        out,
        statistics.fmean(lat),
        statistics.quantiles(lat, n=20)[-1] if len(lat) > 1 else lat[0],
    )


if __name__ == "__main__":
    main()
