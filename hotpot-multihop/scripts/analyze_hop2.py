"""Số liệu nền: tỉ lệ câu bridge mà đoạn hop 2 KHÔNG nằm trong top-K của truy vấn gốc.

Đây là lý do tồn tại của toàn bộ phần đa bước: nếu truy vấn gốc đã kéo được hop 2
lên top-K thì không cần hop thứ hai.
Hop 2 được xác định bằng heuristic trong ``hmr.data.hops`` (câu mơ hồ bị loại, có báo số).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from hmr.common.config import load_config
from hmr.common.io import read_jsonl
from hmr.common.log import setup_logging
from hmr.data.hops import GoldPassage, identify_hop2
from hmr.retrieval.factory import build_retriever

logger = logging.getLogger("analyze_hop2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "train", "all"], default="dev")
    parser.add_argument("--retriever", default=None, help="Mặc định analysis.hop2_retriever")
    parser.add_argument("--depth", type=int, default=None, help="Mặc định analysis.hop2_depth")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config-file", default=None)
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config_file)
    paths = cfg["paths"]
    name = args.retriever or cfg["analysis"]["hop2_retriever"]
    depth = args.depth or int(cfg["analysis"]["hop2_depth"])

    bridge = [
        q
        for q in read_jsonl(paths["questions"])
        if q["type"] == "bridge" and (args.split == "all" or q["split"] == args.split)
    ]
    if args.limit:
        bridge = bridge[: args.limit]
    needed = {d for q in bridge for d in q["gold_doc_ids"]}
    docs = {d["doc_id"]: d for d in read_jsonl(paths["corpus"]) if d["doc_id"] in needed}
    retriever = build_retriever(cfg, name)

    methods: Counter[str] = Counter()
    n_eval = hop2_miss = hop1_miss = any_miss = both_miss = 0
    hop2_ranks: list[int] = []
    for q in tqdm(bridge, desc=f"{name}@{depth}", unit="q"):
        gold = [GoldPassage(d, docs[d]["title"], docs[d]["text"]) for d in q["gold_doc_ids"]]
        hop2, method = identify_hop2(gold, q["answer"])
        methods[method] += 1
        if hop2 is None:
            continue
        hop1 = next(g.doc_id for g in gold if g.doc_id != hop2)
        ranked = [d for d, _ in retriever.search(q["question"], depth)]
        n_eval += 1
        miss2, miss1 = hop2 not in ranked, hop1 not in ranked
        hop2_miss += miss2
        hop1_miss += miss1
        any_miss += miss1 or miss2
        both_miss += miss1 and miss2
        if not miss2:
            hop2_ranks.append(ranked.index(hop2) + 1)

    pct = lambda x: 100 * x / n_eval if n_eval else 0.0  # noqa: E731
    stats = {
        "retriever": name,
        "depth": depth,
        "n_bridge": len(bridge),
        "hop2_method": dict(methods),
        "n_evaluated": n_eval,
        "hop2_not_in_topK_pct": pct(hop2_miss),
        "hop1_not_in_topK_pct": pct(hop1_miss),
        "any_gold_not_in_topK_pct": pct(any_miss),
        "both_not_in_topK_pct": pct(both_miss),
        "median_hop2_rank_when_found": sorted(hop2_ranks)[len(hop2_ranks) // 2] if hop2_ranks else None,
    }
    logger.info(
        "Câu bridge (%s): %d, xác định được hop 2: %d (%s)",
        args.split,
        len(bridge),
        n_eval,
        dict(methods),
    )
    logger.info(">>> Hop 2 KHÔNG nằm trong top-%d của %s: %.2f%%", depth, name, stats["hop2_not_in_topK_pct"])
    logger.info("    (đối chứng) hop 1 không nằm trong top-%d: %.2f%%", depth, stats["hop1_not_in_topK_pct"])
    logger.info("    thiếu ít nhất một đoạn vàng: %.2f%% | thiếu cả hai: %.2f%%", pct(any_miss), pct(both_miss))
    out = Path(paths["runs_dir"]) / f"hop2_analysis_{name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Đã ghi %s", out)


if __name__ == "__main__":
    main()
