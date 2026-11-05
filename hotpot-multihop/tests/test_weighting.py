"""T1: trọng số SMART/BM25 so với cài đặt tham chiếu viết độc lập theo công thức IIR."""

import math
import statistics
from collections import Counter

import pytest

from hmr.index.inverted import InvertedIndex
from hmr.retrieval.bm25 import BM25Retriever
from hmr.retrieval.scoring import ExhaustiveRetriever
from hmr.retrieval.tfidf import TfIdfRetriever
from hmr.retrieval.weighting import BM25Weighting, SmartWeighting, TermWeighting, parse_smart

DOCS = {0: ["apple", "banana", "apple"], 1: ["banana", "cherry"], 2: ["cherry", "durian", "elder", "fig"]}
QUERY = "banana cherry cherry apple unknown"
SCHEMES = ["lnc.ltc", "ltc.ltc", "lnc.lnc", "nnc.ntc", "bnc.btc", "anc.atc", "Lnc.ltc", "nnn.nnn", "lnc.lpc", "apc.ltn"]


def _ref_vector(counts: Counter[str], code: str, n: int, df: dict[str, int]) -> dict[str, float]:
    """Vectơ SMART tính trực tiếp từ bảng 6.15 của IIR."""
    tf_c, df_c, norm_c = code
    max_tf, avg_tf = max(counts.values()), statistics.fmean(counts.values())

    def tf_w(tf: int) -> float:
        return {
            "n": tf,
            "l": 1 + math.log10(tf),
            "a": 0.5 + 0.5 * tf / max_tf,
            "b": 1.0,
            "L": (1 + math.log10(tf)) / (1 + math.log10(avg_tf)),
        }[tf_c]

    def df_w(t: str) -> float:
        return {"n": 1.0, "t": math.log10(n / df[t]), "p": max(0.0, math.log10((n - df[t]) / df[t]))}[df_c]

    v = {t: tf_w(c) * df_w(t) for t, c in counts.items()}
    if norm_c == "c":
        norm = math.sqrt(sum(x * x for x in v.values()))
        v = {t: x / norm for t, x in v.items()} if norm else v
    return v


def _ref_scores(scheme: str) -> dict[int, float]:
    d_code, q_code = scheme.split(".")
    n = len(DOCS)
    df = Counter(t for toks in DOCS.values() for t in set(toks))
    q = _ref_vector(Counter(t for t in QUERY.split() if t in df), q_code, n, df)
    out = {}
    for d, toks in DOCS.items():
        dv = _ref_vector(Counter(toks), d_code, n, df)
        s = sum(w * dv.get(t, 0.0) for t, w in q.items())
        if s:
            out[d] = s
    return out


@pytest.mark.parametrize("scheme", SCHEMES)
def test_smart_matches_reference(toy_index: InvertedIndex, scheme: str) -> None:
    got = dict(TfIdfRetriever(toy_index, str.split, scheme=scheme).search(QUERY, 10))
    ref = _ref_scores(scheme)
    assert got.keys() == ref.keys()
    for d in ref:
        assert got[d] == pytest.approx(ref[d])


@pytest.mark.parametrize("weighting_name", ["bm25", "smart"])
def test_contribution_equals_qw_times_impact(toy_index: InvertedIndex, weighting_name: str) -> None:
    w: TermWeighting = (
        BM25Weighting(toy_index, 1.2, 0.75) if weighting_name == "bm25" else SmartWeighting(toy_index, "anc.atc")
    )
    assert isinstance(w, TermWeighting)
    for term, qw in w.query_weights(QUERY.split()).items():
        impact, contrib = w.impact_fn(term), w.contribution(term, qw)
        for d, tf in toy_index.postings(term):
            assert contrib(d, tf) == pytest.approx(qw * impact(d, tf))


def test_parse_smart() -> None:
    assert parse_smart("lnc.ltc") == ("lnc", "ltc")
    for bad in ["lnc", "lnc.lt", "xnc.ltc", "lnz.ltc"]:
        with pytest.raises(ValueError):
            parse_smart(bad)


def test_exhaustive_stats_and_wrappers(toy_index: InvertedIndex) -> None:
    bm25 = BM25Retriever(toy_index, str.split, k1=1.2, b=0.75)
    generic = ExhaustiveRetriever(toy_index, str.split, BM25Weighting(toy_index, 1.2, 0.75))
    assert bm25.search(QUERY, 10) == generic.search(QUERY, 10)
    # banana(df 2) + cherry(df 2) + apple(df 1) = 5 posting; doc 0,1,2 đều có điểm
    assert (bm25.last_stats.n_terms, bm25.last_stats.postings_scored, bm25.last_stats.candidates) == (3, 5, 3)
    assert bm25.idf("apple") == pytest.approx(math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)))
