import math

import pytest

from hmr.eval.metrics import aggregate, evaluate_per_query, format_table

MEASURES = {"P@5": "P_5", "R@10": "recall_10", "MAP": "map", "nDCG@10": "ndcg_cut_10", "MRR": "recip_rank"}


def test_per_query_values() -> None:
    qrels = {"q1": {"1": 1, "2": 1}, "q2": {"5": 1, "6": 1}}
    run = {"q1": {"1": 3.0, "3": 2.0, "2": 1.0}}  # q2 không có trong run
    pq = evaluate_per_query(run, qrels, ["q1", "q2"], MEASURES)
    assert pq["q1"]["P@5"] == pytest.approx(2 / 5)
    assert pq["q1"]["R@10"] == pytest.approx(1.0)
    assert pq["q1"]["MAP"] == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert pq["q1"]["MRR"] == pytest.approx(1.0)
    idcg = 1 + 1 / math.log2(3)
    assert pq["q1"]["nDCG@10"] == pytest.approx((1 + 1 / math.log2(4)) / idcg)
    assert all(v == 0.0 for v in pq["q2"].values()), "câu không có trong run phải tính 0"


def test_aggregate_by_group() -> None:
    pq = {"a": {"MRR": 1.0}, "b": {"MRR": 0.0}, "c": {"MRR": 0.5}}
    groups = {"a": {"type": "bridge"}, "b": {"type": "bridge"}, "c": {"type": "comparison"}}
    rows = aggregate(pq, groups, ["type"])
    assert rows[0] == ("all", 3, {"MRR": pytest.approx(0.5)})
    assert rows[1] == ("type=bridge", 2, {"MRR": pytest.approx(0.5)})
    assert rows[2] == ("type=comparison", 1, {"MRR": pytest.approx(0.5)})
    assert "type=bridge" in format_table(rows)
