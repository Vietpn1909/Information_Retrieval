"""Bước 1: tải HotpotQA, dựng corpus chung, gióng hàng nhãn vàng.

Đầu ra (theo ``paths`` trong config):
  corpus.jsonl, questions.jsonl, qrels.txt, alignment_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from hmr.common.config import load_config
from hmr.common.io import write_jsonl, write_qrels
from hmr.common.log import setup_logging
from hmr.data.alignment import align_examples
from hmr.data.corpus import Corpus
from hmr.data.hotpot import load_examples

logger = logging.getLogger("build_corpus")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=None, help="YAML cấu hình (mặc định config/default.yaml)")
    parser.add_argument("--n-train", type=int, default=None, help="Ghi đè data.n_train")
    setup_logging()
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    data_cfg = dict(cfg["data"])
    if args.n_train is not None:
        data_cfg["n_train"] = args.n_train
    paths = cfg["paths"]

    examples = list(tqdm(load_examples(data_cfg), desc="Đọc câu hỏi", unit="q"))
    corpus = Corpus()
    for _, ex in tqdm(examples, desc="Dựng corpus", unit="q"):
        corpus.add_example(ex)
    logger.info("Corpus: %d đoạn (đã khử trùng lặp theo title, %d xung đột nội dung)", len(corpus), corpus.n_conflicts)

    records, report = align_examples(examples, corpus, on_error=data_cfg["on_alignment_error"])
    logger.info(
        "Gióng hàng: %d/%d fact (%.2f%%), %d nhiễu nhãn gốc bị loại",
        report.n_aligned,
        report.n_facts - len(report.source_noise),
        100 * report.rate,
        len(report.source_noise),
    )

    corpus.save(paths["corpus"])
    write_jsonl(paths["questions"], records)
    write_qrels(paths["qrels"], {r["qid"]: r["gold_doc_ids"] for r in records})
    report_path = Path(paths["data_dir"]) / "alignment_report.json"
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    n_dev = sum(r["split"] == "dev" for r in records)
    logger.info("Đã ghi %d câu hỏi (%d train, %d dev) vào %s", len(records), len(records) - n_dev, n_dev, paths["data_dir"])


if __name__ == "__main__":
    main()
