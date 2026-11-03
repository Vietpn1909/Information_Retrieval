"""Thí nghiệm truy xuất thưa [M2 – Việt]. Kết quả CSV/JSON + run file ghi vào ``paths.sparse_dir``.

Lệnh con:
  tune      dò lưới k1/b trên train (mục tiêu sparse.tune.objective)
  smart     so sánh các sơ đồ SMART trên dev
  champion  champion list: r × {chấm lại, xấp xỉ} so với BM25 vét cạn
  tiered    chỉ mục phân tầng: fractions × {per_term, global}
  skip      skip pointer: số phép so sánh / thời gian theo bước nhảy
  all       chạy lần lượt tất cả

Ví dụ: python scripts/sparse_experiments.py tune
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

from hmr.common.config import Config, load_config, retriever_config
from hmr.common.interfaces import Retriever
from hmr.common.io import Qrels, read_jsonl, read_qrels, write_trec_run
from hmr.common.log import setup_logging
from hmr.eval.metrics import aggregate, evaluate_per_query
from hmr.index.inverted import InvertedIndex
from hmr.index.skip import IntersectStats, SkipPostings, build_skip_table, intersect, intersect_with_skips, resolve_step
from hmr.retrieval.bm25_grid import evaluate_grid, grid_search, select_best
from hmr.retrieval.champion import ChampionLists, ChampionRetriever
from hmr.retrieval.factory import build_weighting, load_index
from hmr.retrieval.scoring import ExhaustiveRetriever
from hmr.retrieval.tfidf import TfIdfRetriever
from hmr.retrieval.tiered import TieredIndex, TieredRetriever

logger = logging.getLogger("sparse_experiments")

Ranking = list[tuple[int, float]]


# ------------------------------------------------------------------ tiện ích chung
class Context:
    """Dữ liệu dùng chung giữa các lệnh con (nạp lười)."""

    def __init__(self, cfg: Config, limit: int | None) -> None:
        self.cfg = cfg
        self.limit = limit
        self.out_dir = Path(cfg["paths"]["sparse_dir"])
        (self.out_dir / "runs").mkdir(parents=True, exist_ok=True)
        self.depth = int(cfg["sparse"]["depth"])
        self.measures: dict[str, str] = cfg["eval"]["measures"]
        self.qrels: Qrels = read_qrels(cfg["paths"]["qrels"])
        self._questions = list(read_jsonl(cfg["paths"]["questions"]))
        self._indexes: dict[str, InvertedIndex] = {}

    def questions(self, split: str) -> list[dict[str, Any]]:
        qs = [q for q in self._questions if q["split"] == split]
        return qs[: self.limit] if self.limit else qs

    def index(self, name: str) -> InvertedIndex:
        if name not in self._indexes:
            self._indexes[name] = load_index(self.cfg, name)
        return self._indexes[name]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Ghi CSV; cột = hợp các khoá theo thứ tự xuất hiện."""
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Đã ghi %s (%d dòng)", path, len(rows))


def overlap(a: Ranking, b: Ranking, k: int) -> float:
    """|top-k(a) ∩ top-k(b)| / min(k, |top-k(b)|) — b là tham chiếu."""
    ref = {d for d, _ in b[:k]}
    return len({d for d, _ in a[:k]} & ref) / len(ref) if ref else 1.0


def run_config(
    ctx: Context,
    name: str,
    retriever: Retriever,
    questions: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Ranking] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Ranking]]:
    """Chạy một cấu hình: ghi run file, chấm điểm, đo latency/chi phí, overlap với tham chiếu."""
    rankings: dict[str, Ranking] = {}
    lat: list[float] = []
    cost: dict[str, list[float]] = {"postings_scored": [], "candidates": [], "comparisons": [], "tiers_used": []}
    for q in tqdm(questions, desc=name, unit="q", leave=False):
        t0 = time.perf_counter()
        rankings[q["qid"]] = retriever.search(q["question"], ctx.depth)
        lat.append((time.perf_counter() - t0) * 1000)
        st = getattr(retriever, "last_stats", None)
        if st is not None:
            cost["postings_scored"].append(st.postings_scored)
            cost["candidates"].append(st.candidates)
            cost["comparisons"].append(st.comparisons)
        if hasattr(retriever, "last_tiers_used"):
            cost["tiers_used"].append(retriever.last_tiers_used)
    write_trec_run(ctx.out_dir / "runs" / f"{name}.txt", rankings, run_name=name)

    run = {qid: {str(d): s for d, s in r} for qid, r in rankings.items()}
    qids = [q["qid"] for q in questions]
    per_query = evaluate_per_query(run, ctx.qrels, qids, ctx.measures)
    groups = {q["qid"]: {"type": q["type"]} for q in questions}
    table = aggregate(per_query, groups, ["type"])
    row: dict[str, Any] = {"config": name, **(extra or {}), "n": table[0][1]}
    row.update({m: round(v, 4) for m, v in table[0][2].items()})
    for label, _, vals in table[1:]:
        row[f"{label.split('=')[1]}_R@100"] = round(vals["R@100"], 4)
    row["latency_ms_mean"] = round(statistics.fmean(lat), 2)
    row["latency_ms_p95"] = round(statistics.quantiles(lat, n=20)[-1], 2) if len(lat) > 1 else round(lat[0], 2)
    for key, vals in cost.items():
        if vals:
            row[f"{key}_mean"] = round(statistics.fmean(vals), 1)
    if reference is not None:
        for k in (10, 100):
            row[f"overlap@{k}"] = round(statistics.fmean(overlap(rankings[q], reference[q], k) for q in qids), 4)
    logger.info("%s: %s", name, {k: v for k, v in row.items() if k != "config"})
    return row, rankings


def base_setup(ctx: Context, base: str) -> tuple[InvertedIndex, Any, dict[str, Any]]:
    rcfg = retriever_config(ctx.cfg, base)
    index = ctx.index(rcfg["index"])
    return index, build_weighting(index, rcfg), rcfg


# ------------------------------------------------------------------ lệnh con
def cmd_tune(ctx: Context) -> None:
    tcfg = ctx.cfg["sparse"]["tune"]
    rcfg = retriever_config(ctx.cfg, "bm25")
    index = ctx.index(rcfg["index"])
    questions = ctx.questions(tcfg["split"])
    queries = [(q["qid"], q["question"]) for q in questions]
    t0 = time.perf_counter()
    grid = grid_search(index, index.analyze, queries, tcfg["k1"], tcfg["b"], ctx.depth)
    logger.info("Chấm lưới %d cấu hình × %d câu: %.1fs", len(grid.configs()), len(queries), time.perf_counter() - t0)
    rows = evaluate_grid(grid, ctx.qrels, ctx.measures)
    write_csv(ctx.out_dir / "bm25_grid.csv", [{**r, **{m: round(r[m], 4) for m in ctx.measures}} for r in rows])
    best = dict(select_best(rows, tcfg["objective"], tcfg["tiebreak"]))
    default = next((r for r in rows if (r["k1"], r["b"]) == (float(rcfg["k1"]), float(rcfg["b"]))), None)
    if default is None:
        logger.warning("Điểm mặc định (k1=%s, b=%s) không nằm trong lưới", rcfg["k1"], rcfg["b"])

    # Kiểm tra chéo: lưới (numpy) phải trùng BM25 vét cạn tại điểm tốt nhất
    best_idx = grid.configs().index((best["k1"], best["b"]))
    check = ExhaustiveRetriever(index, index.analyze, build_weighting(index, {"type": "bm25", **best}))
    n_check = min(300, len(queries))
    mismatches = sum(
        [d for d, _ in check.search(text, ctx.depth)] != [int(d) for d in grid.top_ids[best_idx, qi] if d >= 0]
        for qi, (_, text) in enumerate(queries[:n_check])
    )
    summary = {
        "split": tcfg["split"],
        "n_queries": len(queries),
        "objective": tcfg["objective"],
        "best": best,
        "default_k1_b": default,
        "grid_vs_retriever_check": {"n_queries": n_check, "mismatches": mismatches},
    }
    (ctx.out_dir / "tune_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Tốt nhất theo %s: k1=%s b=%s | %s", tcfg["objective"], best["k1"], best["b"], best)
    logger.info("Mặc định (k1=%s, b=%s): %s", rcfg["k1"], rcfg["b"], default)
    logger.info("Kiểm tra chéo lưới vs BM25Retriever trên %d câu: %d sai khác", n_check, mismatches)


def cmd_smart(ctx: Context) -> None:
    index = ctx.index(retriever_config(ctx.cfg, "tfidf")["index"])
    questions = ctx.questions("dev")
    rows = []
    for scheme in ctx.cfg["sparse"]["smart_schemes"]:
        row, _ = run_config(ctx, f"smart_{scheme}", TfIdfRetriever(index, index.analyze, scheme), questions, extra={"scheme": scheme})
        rows.append(row)
    for base in ("bm25", "bm25_tuned"):
        idx, w, _ = base_setup(ctx, base)
        row, _ = run_config(ctx, base, ExhaustiveRetriever(idx, idx.analyze, w), questions, extra={"scheme": w.name})
        rows.append(row)
    write_csv(ctx.out_dir / "smart.csv", rows)


def _reference(ctx: Context, base: str, questions: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Ranking]]:
    index, weighting, _ = base_setup(ctx, base)
    return run_config(ctx, f"{base}_exhaustive", ExhaustiveRetriever(index, index.analyze, weighting), questions, extra={"variant": "exhaustive"})


def cmd_champion(ctx: Context) -> None:
    ccfg = ctx.cfg["sparse"]["champion"]
    questions = ctx.questions("dev")
    ref_row, ref = _reference(ctx, ccfg["base"], questions)
    index, weighting, _ = base_setup(ctx, ccfg["base"])
    table = build_skip_table(index)
    t0 = time.perf_counter()
    lists = ChampionLists(index, weighting, r_max=max(ccfg["r"]))
    build_s = round(time.perf_counter() - t0, 1)
    rows = [ref_row]
    for r in ccfg["r"]:
        for rescore in (True, False):
            name = f"champion_r{r}_{'rescore' if rescore else 'approx'}"
            retriever = ChampionRetriever(index, index.analyze, weighting, lists, r, table, rescore)
            row, _ = run_config(ctx, name, retriever, questions, ref, {"variant": "champion", "r": r, "rescore": rescore, "build_s": build_s})
            rows.append(row)
    write_csv(ctx.out_dir / "champion.csv", rows)


def cmd_tiered(ctx: Context) -> None:
    tcfg = ctx.cfg["sparse"]["tiered"]
    questions = ctx.questions("dev")
    ref_row, ref = _reference(ctx, tcfg["base"], questions)
    index, weighting, _ = base_setup(ctx, tcfg["base"])
    table = build_skip_table(index)
    rows = [ref_row]
    for mode in tcfg["modes"]:
        for fractions in tcfg["fractions"]:
            t0 = time.perf_counter()
            tiers = TieredIndex(index, weighting, fractions, mode)
            build_s = round(time.perf_counter() - t0, 1)
            for min_cand in tcfg["min_candidates"]:
                name = f"tiered_{mode}_{'-'.join(str(f) for f in fractions)}_mc{min_cand or 'k'}"
                retriever = TieredRetriever(index, index.analyze, weighting, tiers, table, min_candidates=min_cand)
                extra = {"variant": "tiered", "mode": mode, "fractions": str(fractions),
                         "min_candidates": min_cand or "k", "build_s": build_s}
                row, _ = run_config(ctx, name, retriever, questions, ref, extra)
                rows.append(row)
    write_csv(ctx.out_dir / "tiered.csv", rows)


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def cmd_skip(ctx: Context) -> None:
    scfg = ctx.cfg["sparse"]["skip"]
    index, weighting, _ = base_setup(ctx, ctx.cfg["sparse"]["champion"]["base"])
    questions = ctx.questions("dev")
    rng = random.Random(int(scfg["seed"]))

    # Kịch bản 1: giao hai postings (truy vấn AND hai term) lấy từ câu hỏi thật
    pairs: set[tuple[str, str]] = set()
    while len(pairs) < int(scfg["n_pairs"]):
        terms = sorted({t for t in index.analyze(rng.choice(questions)["question"]) if index.df(t) > 0})
        if len(terms) >= 2:
            pairs.add(tuple(rng.sample(terms, 2)))  # type: ignore[arg-type]
    lists = {t: [d for d, _ in index.postings(t)] for p in pairs for t in p}
    rows: list[dict[str, Any]] = []
    plain = IntersectStats()
    truth, t_plain = _timed(lambda: [intersect(lists[a], lists[b], plain) for a, b in pairs])
    rows.append({"scenario": "AND 2 term", "step": "none", "comparisons": plain.comparisons, "skips": 0, "time_ms": round(t_plain, 1), "mismatches": 0})
    for spec in scfg["steps"]:
        st = IntersectStats()
        out, t = _timed(lambda: [
            intersect_with_skips(lists[a], lists[b], resolve_step(spec, len(lists[a])), resolve_step(spec, len(lists[b])), st)
            for a, b in pairs
        ])
        rows.append({"scenario": "AND 2 term", "step": spec, "comparisons": st.comparisons, "skips": st.skips,
                     "time_ms": round(t, 1), "mismatches": sum(x != y for x, y in zip(out, truth))})

    # Kịch bản 2: tra tf của ứng viên (top-1000 BM25) trong postings đầy đủ — như bước chấm lại ở T4/T5
    exhaustive = ExhaustiveRetriever(index, index.analyze, weighting)
    jobs: list[tuple[str, list[int]]] = []
    for q in tqdm(questions[: int(scfg["n_pairs"]) // 4], desc="Ứng viên top-1000", unit="q", leave=False):
        cands = sorted(d for d, _ in exhaustive.search(q["question"], 1000))
        jobs += [(t, cands) for t in set(index.analyze(q["question"])) if index.df(t) > 0]
    full = {t: index.postings(t) for t, _ in jobs}
    truth2 = [[dict(full[t]).get(c, 0) for c in cands] for t, cands in jobs]
    for spec in ["none", *scfg["steps"]]:
        table = {t: SkipPostings.from_postings(full[t], 1 if spec == "none" else spec) for t in full}
        st = IntersectStats()
        out, t = _timed(lambda: [table[term].lookup(cands, st) for term, cands in jobs])
        rows.append({"scenario": "tra ứng viên top-1000", "step": spec, "comparisons": st.comparisons, "skips": st.skips,
                     "time_ms": round(t, 1), "mismatches": sum(x != y for x, y in zip(out, truth2))})
    base_cmp = {s: next(r["comparisons"] for r in rows if r["scenario"] == s and r["step"] == "none") for s in {r["scenario"] for r in rows}}
    for r in rows:
        r["comparisons_vs_none"] = round(r["comparisons"] / base_cmp[r["scenario"]], 4)
    write_csv(ctx.out_dir / "skip.csv", rows)
    logger.info("Skip pointer (%d cặp, %d lần tra):\n%s", len(pairs), len(jobs),
                "\n".join(f"  {r['scenario']:<24} step={r['step']!s:<5} so sánh={r['comparisons']:>12,} ({r['comparisons_vs_none']:.1%}) "
                          f"thời gian={r['time_ms']:>9.1f}ms sai={r['mismatches']}" for r in rows))


COMMANDS: dict[str, Callable[[Context], None]] = {
    "tune": cmd_tune, "smart": cmd_smart, "champion": cmd_champion, "tiered": cmd_tiered, "skip": cmd_skip,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu mỗi split (thử nhanh)")
    parser.add_argument("--config-file", default=None)
    setup_logging()
    args = parser.parse_args()
    ctx = Context(load_config(args.config_file), args.limit)
    for name in COMMANDS if args.command == "all" else [args.command]:
        t0 = time.perf_counter()
        logger.info("=== %s ===", name)
        COMMANDS[name](ctx)
        logger.info("=== %s xong sau %.1f phút ===", name, (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
