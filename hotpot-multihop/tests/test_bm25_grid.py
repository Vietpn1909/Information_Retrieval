"""T3: lưới k1/b vectơ hoá phải cho top-k trùng BM25Retriever."""

from hmr.index.inverted import InvertedIndex
from hmr.retrieval.bm25 import BM25Retriever
from hmr.retrieval.bm25_grid import evaluate_grid, grid_search, select_best

QUERIES = [("q1", "banana cherry cherry apple"), ("q2", "fig apple"), ("q3", "cherry"), ("q4", "unknown")]
K1S, BS = [0.3, 1.2, 2.4], [0.0, 0.75, 1.0]


def test_grid_matches_bm25_retriever(toy_index: InvertedIndex) -> None:
    res = grid_search(toy_index, str.split, QUERIES, K1S, BS, depth=2, show_progress=False)
    assert res.top_ids.shape == (9, 4, 2)
    for ci, (k1, b) in enumerate(res.configs()):
        bm25 = BM25Retriever(toy_index, str.split, k1=k1, b=b)
        for qi, (_, text) in enumerate(QUERIES):
            expected = [d for d, _ in bm25.search(text, 2)]
            got = [int(d) for d in res.top_ids[ci, qi] if d >= 0]
            assert got == expected, (k1, b, text)


def test_evaluate_and_select(toy_index: InvertedIndex) -> None:
    res = grid_search(toy_index, str.split, QUERIES[:2], [1.2], [0.0, 0.75], depth=3, show_progress=False)
    qrels = {"q1": {"1": 1}, "q2": {"2": 1}}
    rows = evaluate_grid(res, qrels, {"R@1": "recall_1", "MRR": "recip_rank"}, show_progress=False)
    assert [(r["k1"], r["b"]) for r in rows] == [(1.2, 0.0), (1.2, 0.75)]
    assert all(0.0 <= r["MRR"] <= 1.0 and r["n"] == 2 for r in rows)
    run = res.run(0)
    assert list(run["q1"].values()) == sorted(run["q1"].values(), reverse=True)
    best = select_best([{"R": 0.5, "M": 0.1}, {"R": 0.5, "M": 0.3}, {"R": 0.4, "M": 0.9}], "R", "M")
    assert best == {"R": 0.5, "M": 0.3}
