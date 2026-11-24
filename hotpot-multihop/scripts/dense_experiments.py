"""Thí nghiệm truy xuất dày & lai [M3 – Khiêm]. Bảng kết quả ghi vào ``paths.dense_runs``.

Lệnh con:
  ann          FAISS chính xác (flat) vs xấp xỉ (HNSW theo efSearch, IVF theo nprobe) trên dev
  tune         dò cách gộp hybrid (RRF: rrf_k × trọng số; linear: chuẩn hoá × alpha) trên train
  rerank_tune  dò rerank_depth của cross-encoder trên một phần train
  eval         bảng chính trên dev: BM25 / dense / hybrid / rerank + phân tích hop 2
  all          chạy lần lượt tất cả

Yêu cầu: đã chạy build_corpus.py, build_index.py --index stem, build_dense.py --model bge_small minilm.
Ví dụ: uv run python scripts/dense_experiments.py eval   (--limit 200 để chạy thử nhanh)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from hmr.common.config import Config, load_config, retriever_config
from hmr.common.io import Qrels, read_jsonl, read_qrels, write_trec_run
from hmr.common.log import setup_logging
from hmr.data.hops import GoldPassage, identify_hop2
from hmr.eval.metrics import aggregate, evaluate_per_query
from hmr.retrieval.dense import (
    DenseRetriever,
    build_faiss,
    configure_threads,
    load_embeddings,
    make_encoder,
    set_search_param,
)
from hmr.retrieval.factory import build_retriever
from hmr.retrieval.hybrid import HybridRetriever, fuse
from hmr.retrieval.rerank import RerankRetriever, append_tail, load_passage_texts, make_scorer, order_head, rerank_ranking

logger = logging.getLogger("dense_experiments")

Ranking = list[tuple[int, float]]
Rankings = dict[str, Ranking]  # qid -> ranking


# ------------------------------------------------------------------ tiện ích chung
def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Ghi CSV; cột = hợp các khoá theo thứ tự xuất hiện."""
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Đã ghi %s (%d dòng)", path, len(rows))


def overlap(a: Ranking, b: Ranking, k: int) -> float:
    """|top-k(a) ∩ top-k(b)| / |top-k(b)| — b là tham chiếu."""
    ref = {d for d, _ in b[:k]}
    return len({d for d, _ in a[:k]} & ref) / len(ref) if ref else 1.0


def both_gold_at(ranking: Ranking, gold: Sequence[int], k: int) -> bool:
    """Cả hai đoạn vàng đều nằm trong top-k (điều kiện cần để trả lời câu multi-hop)."""
    top = {d for d, _ in ranking[:k]}
    return all(g in top for g in gold)


class Context:
    """Dữ liệu dùng chung giữa các lệnh con, nạp lười và có cache."""

    def __init__(self, cfg: Config, limit: int | None) -> None:
        self.cfg = cfg
        self.limit = limit
        self.ecfg: dict[str, Any] = cfg["dense"]["experiments"]
        self.out_dir = Path(cfg["paths"]["dense_runs"])
        self.run_dir = self.out_dir / "runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.depth = int(self.ecfg["depth"])
        self.cand_depth = int(self.ecfg["candidate_depth"])
        self.measures: dict[str, str] = cfg["eval"]["measures"]
        self.qrels: Qrels = read_qrels(cfg["paths"]["qrels"])
        self._questions = list(read_jsonl(cfg["paths"]["questions"]))
        self._retrievers: dict[str, Any] = {}
        self._texts: list[str] | None = None
        self._scorers: dict[str, Any] = {}
        # cache xếp hạng: (tên cấu hình, split) -> {qid: ranking} ở độ sâu cand_depth
        self._cache: dict[tuple[str, str], Rankings] = {}

    def questions(self, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Câu hỏi của một split (cắt theo ``--limit`` và ``limit``)."""
        qs = [q for q in self._questions if q["split"] == split]
        for lim in (self.limit, limit):
            if lim:
                qs = qs[:lim]
        return qs

    def retriever(self, name: str) -> Any:
        """Bộ truy xuất theo tên cấu hình (có cache).

        Hybrid / rerank được lắp từ các thành phần đã cache — không nạp lại model hay chỉ mục
        (tiết kiệm bộ nhớ GPU), kết quả giống hệt ``build_retriever``.
        """
        if name not in self._retrievers:
            rcfg = retriever_config(self.cfg, name)
            if rcfg["type"] == "hybrid":
                comps = [self.retriever(c) for c in rcfg["components"]]
                self._retrievers[name] = HybridRetriever(comps, rcfg, int(rcfg.get("depth", 200)))
            elif rcfg["type"] == "rerank":
                self._retrievers[name] = RerankRetriever(
                    self.retriever(rcfg["base"]),
                    self.scorer(rcfg["model"]),
                    self.texts(),
                    int(rcfg["rerank_depth"]),
                    rcfg.get("combine"),
                )
            else:
                self._retrievers[name] = build_retriever(self.cfg, name)
        return self._retrievers[name]

    def texts(self) -> list[str]:
        """Văn bản các đoạn theo doc_id (cho cross-encoder)."""
        if self._texts is None:
            self._texts = load_passage_texts(self.cfg)
        return self._texts

    def scorer(self, key: str) -> Any:
        """Cross-encoder theo khoá (có cache)."""
        if key not in self._scorers:
            self._scorers[key] = make_scorer(self.cfg, key)
        return self._scorers[key]

    # ---------------------------------------------------------------- xếp hạng có cache
    def rankings(self, name: str, questions: Sequence[Mapping[str, Any]], split: str) -> Rankings:
        """Xếp hạng (độ sâu ``cand_depth``) của cấu hình ``name`` cho các câu hỏi.

        Hybrid gộp lại từ cache của các thành phần; rerank chấm lại trên cache của base —
        mỗi mô hình chỉ chạy một lần cho mỗi câu hỏi dù xuất hiện trong nhiều cấu hình.
        """
        key = (name, split)
        cached = self._cache.get(key, {})
        todo = [q for q in questions if q["qid"] not in cached]
        if todo:
            rcfg = retriever_config(self.cfg, name)
            kind = rcfg["type"]
            texts = [q["question"] for q in todo]
            if kind == "hybrid":
                comps = [self.rankings(c, todo, split) for c in rcfg["components"]]
                new = [fuse([c[q["qid"]][: int(rcfg.get("depth", 200))] for c in comps], rcfg, self.cand_depth) for q in todo]
            elif kind == "rerank":
                base = self.rankings(rcfg["base"], todo, split)
                scorer = self.scorer(rcfg["model"])
                d = int(rcfg["rerank_depth"])
                comb = rcfg.get("combine")
                new = [
                    rerank_ranking(q["question"], base[q["qid"]], self.texts(), scorer, d, self.cand_depth, comb)
                    for q in tqdm(todo, desc=name, unit="q", leave=False)
                ]
            elif kind == "dense":
                new = self.retriever(name).search_batch(texts, self.cand_depth)
            else:
                r = self.retriever(name)
                new = [r.search(t, self.cand_depth) for t in tqdm(texts, desc=name, unit="q", leave=False)]
            cached.update({q["qid"]: rk for q, rk in zip(todo, new)})
            self._cache[key] = cached
        return {q["qid"]: cached[q["qid"]] for q in questions}

    def evaluate(self, rankings: Mapping[str, Ranking], questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Điểm tổng + R@100/nDCG@10 theo type, xếp hạng cắt ở ``depth``."""
        run = {qid: {str(d): s for d, s in r[: self.depth]} for qid, r in rankings.items()}
        qids = [q["qid"] for q in questions]
        per_query = evaluate_per_query(run, self.qrels, qids, self.measures)
        table = aggregate(per_query, {q["qid"]: {"type": q["type"]} for q in questions}, ["type"])
        row: dict[str, Any] = {"n": table[0][1]}
        row.update({m: round(v, 4) for m, v in table[0][2].items()})
        for label, _, vals in table[1:]:
            t = label.split("=")[1]
            row[f"{t}_R@100"] = round(vals["R@100"], 4)
            row[f"{t}_nDCG@10"] = round(vals["nDCG@10"], 4)
        return row

    def write_run(self, name: str, rankings: Mapping[str, Ranking]) -> None:
        """Ghi TREC run (không commit — bị .gitignore) để Vinh / evaluate.py dùng lại."""
        write_trec_run(self.run_dir / f"{name}.txt", {q: r[: self.depth] for q, r in rankings.items()}, run_name=name)


def better(a: Mapping[str, Any], b: Mapping[str, Any] | None, objective: str, tiebreak: str) -> bool:
    """a tốt hơn b theo (objective, tiebreak)."""
    return b is None or (a[objective], a[tiebreak]) > (b[objective], b[tiebreak])


# ------------------------------------------------------------------ ann
def cmd_ann(ctx: Context) -> None:
    """Flat vs HNSW vs IVF: chất lượng, độ trùng với flat, thời gian tìm, thời gian dựng."""
    cfg = ctx.cfg
    rcfg = retriever_config(cfg, "dense_bge")
    key = rcfg["model"]
    questions = ctx.questions("dev")
    qids = [q["qid"] for q in questions]
    emb = load_embeddings(cfg, key)
    encoder = make_encoder(cfg, key, "query")

    t0 = time.perf_counter()
    qvecs = encoder([q["question"] for q in questions])
    enc_batch_ms = (time.perf_counter() - t0) * 1000 / len(questions)
    sample = [q["question"] for q in questions[:200]]
    t0 = time.perf_counter()
    for s in sample:
        encoder([s])
    enc_single_ms = (time.perf_counter() - t0) * 1000 / len(sample)
    logger.info("Mã hoá truy vấn: %.2f ms/câu (lô), %.2f ms/câu (từng câu)", enc_batch_ms, enc_single_ms)

    variants: list[tuple[str, dict[str, Any], str | None, list[int]]] = [
        ("flat", {"kind": "flat"}, None, [0]),
        ("hnsw", retriever_config(cfg, "dense_bge_hnsw")["faiss"], "hnsw", ctx.ecfg["ann"]["hnsw_ef_search"]),
        ("ivf", retriever_config(cfg, "dense_bge_ivf")["faiss"], "ivf", ctx.ecfg["ann"]["ivf_nprobe"]),
    ]
    rows: list[dict[str, Any]] = []
    reference: Rankings | None = None
    for label, spec, kind, values in variants:
        t0 = time.perf_counter()
        index = build_faiss(emb, spec)
        build_s = time.perf_counter() - t0
        retr = DenseRetriever(encoder, index)
        for v in values:
            if kind:
                set_search_param(index, kind, v)
            # Đo từng truy vấn một (giống tình huống phục vụ thật), vectơ đã mã hoá sẵn
            res: list[Ranking] = []
            lat: list[float] = []
            for i in range(len(questions)):
                t0 = time.perf_counter()
                res += retr.search_vectors(qvecs[i : i + 1], ctx.depth)
                lat.append((time.perf_counter() - t0) * 1000)
            rankings = dict(zip(qids, res))
            if reference is None:
                reference = rankings
            row = {
                "index": label,
                "param": {"hnsw": "efSearch", "ivf": "nprobe"}.get(label, "-"),
                "value": v if kind else "-",
                "build_s": round(build_s, 2),
                **ctx.evaluate(rankings, questions),
                "overlap@10_vs_flat": round(statistics.fmean(overlap(rankings[q], reference[q], 10) for q in qids), 4),
                "overlap@100_vs_flat": round(statistics.fmean(overlap(rankings[q], reference[q], 100) for q in qids), 4),
                "search_ms_mean": round(statistics.fmean(lat), 3),
                "search_ms_p95": round(statistics.quantiles(lat, n=20)[-1], 3) if len(lat) > 1 else round(lat[0], 3),
                "query_encode_ms": round(enc_single_ms, 2),
            }
            rows.append(row)
            logger.info("%s", row)
    write_csv(ctx.out_dir / "ann.csv", rows)


# ------------------------------------------------------------------ tune
def cmd_tune(ctx: Context) -> None:
    """Dò RRF (rrf_k × weights) và linear (norm × alpha) trên train cho hybrid bm25_tuned + dense_bge."""
    ecfg = ctx.ecfg
    split = ecfg["tune_split"]
    questions = ctx.questions(split)
    base = retriever_config(ctx.cfg, "hybrid_rrf")
    comps = [ctx.rankings(c, questions, split) for c in base["components"]]
    depth = int(base.get("depth", 200))
    obj, tie = ecfg["objective"], ecfg["tiebreak"]

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(base["components"]):
        rows.append({"fusion": f"only:{name}", **ctx.evaluate({q: comps[i][q] for q in comps[i]}, questions)})

    def run(params: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
        fused = {q["qid"]: fuse([c[q["qid"]][:depth] for c in comps], params, ctx.depth) for q in questions}
        row = {**label, **ctx.evaluate(fused, questions)}
        rows.append(row)
        return row

    best: dict[str, dict[str, Any] | None] = {"rrf": None, "linear": None}
    for rrf_k in tqdm(ecfg["rrf_k"], desc="rrf", unit="cfg"):
        for w in ecfg["rrf_weights"]:
            row = run({"fusion": "rrf", "rrf_k": rrf_k, "weights": w}, {"fusion": "rrf", "rrf_k": rrf_k, "w_sparse": w[0], "w_dense": w[1]})
            if better(row, best["rrf"], obj, tie):
                best["rrf"] = row
    for norm in ecfg["norms"]:
        for alpha in tqdm(ecfg["alpha"], desc=f"linear/{norm}", unit="cfg"):
            row = run({"fusion": "linear", "norm": norm, "alpha": alpha}, {"fusion": "linear", "norm": norm, "alpha": alpha})
            if better(row, best["linear"], obj, tie):
                best["linear"] = row
    write_csv(ctx.out_dir / "hybrid_tune.csv", rows)
    summary = {"split": split, "n_queries": len(questions), "objective": obj, "tiebreak": tie, "best": best}
    (ctx.out_dir / "hybrid_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Tốt nhất: %s", json.dumps(best, indent=1))


# ------------------------------------------------------------------ rerank_tune
def cmd_rerank_tune(ctx: Context) -> None:
    """Dò reranker × rerank_depth × cách gộp trên train, cho từng base.

    Điểm cross-encoder của một cặp (câu hỏi, đoạn) không phụ thuộc depth hay cách gộp, nên mỗi
    (base, reranker) chỉ chấm một lần ở depth lớn nhất rồi suy ra mọi tổ hợp còn lại.
    """
    ecfg = ctx.ecfg
    split = ecfg["tune_split"]
    questions = ctx.questions(split, int(ecfg["rerank_tune_questions"]))
    depths = sorted(int(d) for d in ecfg["rerank_depths"])
    combines: list[dict[str, Any] | None] = [None, *ecfg["rerank_combine"]]
    obj, tie = ecfg["rerank_objective"], ecfg["objective"]
    texts = ctx.texts()
    rows: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    for base_name in ecfg["rerank_bases"]:
        base = ctx.rankings(base_name, questions, split)
        base_row = {"base": base_name, "reranker": "-", "rerank_depth": 0, "combine": "-", **ctx.evaluate(base, questions)}
        rows.append(base_row)
        best[base_name] = base_row
        for model in ecfg["rerank_models"]:
            scorer = ctx.scorer(model)
            ce: dict[str, list[float]] = {}
            t0 = time.perf_counter()
            for q in tqdm(questions, desc=f"{base_name}/{model}", unit="q"):
                head = base[q["qid"]][: depths[-1]]
                ce[q["qid"]] = list(scorer([(q["question"], texts[d]) for d, _ in head]))
            ms_per_pair = (time.perf_counter() - t0) * 1000 / (len(questions) * depths[-1])
            for depth in depths:
                for comb in combines:
                    reranked = {
                        q["qid"]: append_tail(
                            order_head(base[q["qid"]][:depth], ce[q["qid"]][:depth], comb), base[q["qid"]][depth:]
                        )
                        for q in questions
                    }
                    row = {
                        "base": base_name,
                        "reranker": model,
                        "rerank_depth": depth,
                        "combine": json.dumps(comb) if comb else "-",
                        **ctx.evaluate(reranked, questions),
                        "rerank_ms_per_q": round(ms_per_pair * depth, 2),
                    }
                    rows.append(row)
                    if better(row, best[base_name], obj, tie):
                        best[base_name] = row
        logger.info("Tốt nhất cho %s: %s", base_name, best[base_name])
    write_csv(ctx.out_dir / "rerank_tune.csv", rows)
    summary = {"split": split, "n_queries": len(questions), "objective": obj, "best": best}
    (ctx.out_dir / "rerank_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ eval
def hop2_stats(rankings: Rankings, questions: Sequence[Mapping[str, Any]], docs: Mapping[int, Mapping[str, Any]], k: int) -> dict[str, Any]:
    """Tỉ lệ câu bridge có đoạn hop 2 / hop 1 nằm ngoài top-k (giống analyze_hop2.py)."""
    n = miss2 = miss1 = 0
    for q in questions:
        if q["type"] != "bridge":
            continue
        gold = [GoldPassage(d, docs[d]["title"], docs[d]["text"]) for d in q["gold_doc_ids"]]
        hop2, _ = identify_hop2(gold, q["answer"])
        if hop2 is None:
            continue
        hop1 = next(g.doc_id for g in gold if g.doc_id != hop2)
        top = {d for d, _ in rankings[q["qid"]][:k]}
        n += 1
        miss2 += hop2 not in top
        miss1 += hop1 not in top
    return {
        "bridge_evaluated": n,
        f"hop2_miss@{k}_pct": round(100 * miss2 / n, 2) if n else None,
        f"hop1_miss@{k}_pct": round(100 * miss1 / n, 2) if n else None,
    }


def measure_latency(ctx: Context, name: str, questions: Sequence[Mapping[str, Any]], n: int) -> tuple[float, float]:
    """Latency từng câu (ms, mean và p95) qua đúng ``Retriever.search`` như run_retrieval.py."""
    retr = ctx.retriever(name)
    lat = []
    for q in questions[:n]:
        t0 = time.perf_counter()
        retr.search(q["question"], ctx.depth)
        lat.append((time.perf_counter() - t0) * 1000)
    p95 = statistics.quantiles(lat, n=20)[-1] if len(lat) > 1 else lat[0]
    return round(statistics.fmean(lat), 1), round(p95, 1)


def cmd_eval(ctx: Context) -> None:
    """Bảng chính trên dev."""
    configs = ctx.ecfg.get("eval_configs") or [  # mặc định nếu config không ghi
        "bm25_tuned",
        "dense_minilm",
        "dense_bge",
        "hybrid_linear",
        "hybrid_rrf",
        "rerank_bm25",
        "rerank_hybrid",
    ]
    questions = ctx.questions("dev")
    needed = {d for q in questions for d in q["gold_doc_ids"]}
    docs = {d["doc_id"]: d for d in read_jsonl(ctx.cfg["paths"]["corpus"]) if d["doc_id"] in needed}
    n_lat = int(ctx.ecfg.get("latency_questions", 200))
    rows: list[dict[str, Any]] = []
    for name in configs:
        t0 = time.perf_counter()
        rankings = ctx.rankings(name, questions, "dev")
        logger.info("%s: %d câu trong %.1fs", name, len(rankings), time.perf_counter() - t0)
        ctx.write_run(name, rankings)
        row: dict[str, Any] = {"config": name, "type": retriever_config(ctx.cfg, name)["type"], **ctx.evaluate(rankings, questions)}
        for k in ctx.ecfg["both_gold_k"]:
            row[f"both_gold@{k}"] = round(statistics.fmean(both_gold_at(rankings[q["qid"]], q["gold_doc_ids"], k) for q in questions), 4)
        row.update(hop2_stats(rankings, questions, docs, ctx.depth))
        row["latency_ms_mean"], row["latency_ms_p95"] = measure_latency(ctx, name, questions, n_lat)
        rows.append(row)
        logger.info("%s", row)
    write_csv(ctx.out_dir / "dev.csv", rows)
    # độ trùng top-100 giữa BM25 và dense: hai bộ "bổ sung" nhau tới đâu
    if {"bm25_tuned", "dense_bge"} <= set(configs):
        bm, de = ctx.rankings("bm25_tuned", questions, "dev"), ctx.rankings("dense_bge", questions, "dev")
        comp = {
            "overlap@10": round(statistics.fmean(overlap(de[q["qid"]], bm[q["qid"]], 10) for q in questions), 4),
            "overlap@100": round(statistics.fmean(overlap(de[q["qid"]], bm[q["qid"]], 100) for q in questions), 4),
        }
        for k in (10, 100):
            only_bm = only_de = both = neither = 0
            for q in questions:
                for g in q["gold_doc_ids"]:
                    in_bm = g in {d for d, _ in bm[q["qid"]][:k]}
                    in_de = g in {d for d, _ in de[q["qid"]][:k]}
                    both += in_bm and in_de
                    only_bm += in_bm and not in_de
                    only_de += in_de and not in_bm
                    neither += not in_bm and not in_de
            total = 2 * len(questions)
            comp[f"gold@{k}"] = {
                "both": round(both / total, 4),
                "only_bm25": round(only_bm / total, 4),
                "only_dense": round(only_de / total, 4),
                "neither": round(neither / total, 4),
            }
        (ctx.out_dir / "complementarity.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
        logger.info("BM25 vs dense: %s", comp)


COMMANDS = {"ann": cmd_ann, "tune": cmd_tune, "rerank_tune": cmd_rerank_tune, "eval": cmd_eval}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu mỗi split (thử nhanh)")
    parser.add_argument("--config-file", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config_file)
    configure_threads(cfg)
    np.random.seed(0)
    ctx = Context(cfg, args.limit)
    for name in COMMANDS if args.command == "all" else [args.command]:
        t0 = time.perf_counter()
        COMMANDS[name](ctx)
        logger.info("== %s xong trong %.1f phút", name, (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
