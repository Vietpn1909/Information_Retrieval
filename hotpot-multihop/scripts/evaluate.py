"""Bước 4: chấm điểm một TREC run file: P@5, R@10, MAP, nDCG@10, MRR, tách theo type/level.

Ví dụ: python scripts/evaluate.py --run runs/bm25.txt --qrels data/qrels.txt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hmr.common.config import load_config
from hmr.common.io import read_jsonl, read_qrels, read_trec_run
from hmr.common.log import setup_logging
from hmr.eval.metrics import aggregate, evaluate_per_query, format_table

logger = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--qrels", default=None, help="Mặc định paths.qrels")
    parser.add_argument("--questions", default=None, help="Mặc định paths.questions (lấy type/level)")
    parser.add_argument("--split", choices=["dev", "train", "all"], default="dev", help="Tập câu hỏi được chấm")
    parser.add_argument("--json-out", default=None, help="Ghi kết quả tổng hợp ra JSON")
    parser.add_argument("--config-file", default=None, help="YAML cấu hình (mặc định config/default.yaml)")
    setup_logging()
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    qrels = read_qrels(args.qrels or cfg["paths"]["qrels"])
    questions = [
        q for q in read_jsonl(args.questions or cfg["paths"]["questions"]) if args.split == "all" or q["split"] == args.split
    ]
    run = read_trec_run(args.run)
    qids = [q["qid"] for q in questions]
    missing = sum(q not in run for q in qids)
    if missing:
        logger.warning("%d/%d câu hỏi không có trong run -> tính 0 điểm", missing, len(qids))

    group_by = cfg["eval"]["group_by"]
    per_query = evaluate_per_query(run, qrels, qids, cfg["eval"]["measures"])
    groups = {q["qid"]: {k: q[k] for k in group_by} for q in questions}
    rows = aggregate(per_query, groups, group_by)
    logger.info("Kết quả %s (split=%s):\n%s", Path(args.run).name, args.split, format_table(rows))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([{"group": g, "n": n, **v} for g, n, v in rows], indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
