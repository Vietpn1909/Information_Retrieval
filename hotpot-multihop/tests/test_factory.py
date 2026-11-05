"""T6: dựng mọi cấu hình truy xuất thưa từ YAML (vét cạn, SMART, champion, tiered)."""

import sys
import types
from pathlib import Path

import pytest
import yaml

from hmr.common.config import load_config
from hmr.common.interfaces import RetrievalResult
from hmr.index.inverted import InvertedIndex
from hmr.retrieval import factory
from hmr.retrieval.champion import ChampionRetriever
from hmr.retrieval.factory import SingleHopAdapter, build_retriever, build_system
from hmr.retrieval.scoring import ExhaustiveRetriever
from hmr.retrieval.tiered import TieredRetriever

BM25 = {"type": "bm25", "index": "toy", "k1": 0.9, "b": 1.0}


@pytest.fixture
def cfg(tmp_path: Path, toy_index: InvertedIndex) -> dict:
    (tmp_path / "config").mkdir()
    toy_index.save(tmp_path / "idx" / "toy.pkl")
    raw = {
        "paths": {"index_dir": "idx"},
        "retrievers": {
            "bm25": BM25,
            "smart": {"type": "tfidf", "index": "toy", "scheme": "anc.atc"},
            "champ": {**BM25, "pruning": {"kind": "champion", "r": 100, "rescore": True}},
            "champ_approx": {**BM25, "pruning": {"kind": "champion", "r": 1, "rescore": False}},
            "tiers": {**BM25, "pruning": {"kind": "tiered", "fractions": [0.5, 0.5], "mode": "global", "min_candidates": 99}},
            "bad": {**BM25, "pruning": {"kind": "wand"}},
        },
    }
    path = tmp_path / "config" / "test.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def test_build_all_kinds(cfg: dict) -> None:
    exhaustive = build_retriever(cfg, "bm25")
    assert type(exhaustive) is ExhaustiveRetriever
    assert build_retriever(cfg, "smart").weighting.name == "anc.atc"
    champ, tiers = build_retriever(cfg, "champ"), build_retriever(cfg, "tiers")
    assert isinstance(champ, ChampionRetriever) and champ.r == 100 and champ.rescore
    assert isinstance(tiers, TieredRetriever) and tiers.min_candidates == 99
    for q in ["banana cherry", "apple fig", "durian"]:
        # r ≥ df và mọi tầng được duyệt ⇒ trùng vét cạn
        assert champ.search(q, 5) == tiers.search(q, 5) == exhaustive.search(q, 5)
    approx = build_retriever(cfg, "champ_approx")
    assert approx.skip_table is None and len(approx.search("banana cherry", 5)) <= 2


def test_unknown_pruning(cfg: dict) -> None:
    with pytest.raises(ValueError):
        build_retriever(cfg, "bad")


# ------------------------------------------------------------------ chống conflict
def test_config_parts_are_merged(tmp_path: Path) -> None:
    (tmp_path / "config" / "parts").mkdir(parents=True)
    main = {"paths": {"runs_dir": "runs"}, "retrievers": {"bm25": BM25}}
    (tmp_path / "config" / "default.yaml").write_text(yaml.safe_dump(main), encoding="utf-8")
    part = {"paths": {"dense_dir": "data/dense"}, "retrievers": {"dense_bge": {"type": "dense"}}, "dense": {"dim": 384}}
    (tmp_path / "config" / "parts" / "dense.yaml").write_text(yaml.safe_dump(part), encoding="utf-8")
    cfg = load_config(tmp_path / "config" / "default.yaml")
    assert set(cfg["retrievers"]) == {"bm25", "dense_bge"} and cfg["dense"] == {"dim": 384}
    assert Path(cfg["paths"]["dense_dir"]) == (tmp_path / "data" / "dense").resolve()


@pytest.mark.parametrize("clash", [{"retrievers": {"bm25": BM25}}, {"paths": "x"}])
def test_config_part_clash_raises(tmp_path: Path, clash: dict) -> None:
    (tmp_path / "config" / "parts").mkdir(parents=True)
    (tmp_path / "config" / "default.yaml").write_text(
        yaml.safe_dump({"paths": {"runs_dir": "runs"}, "retrievers": {"bm25": BM25}}), encoding="utf-8"
    )
    (tmp_path / "config" / "parts" / "x.yaml").write_text(yaml.safe_dump(clash), encoding="utf-8")
    with pytest.raises(ValueError, match="x.yaml"):
        load_config(tmp_path / "config" / "default.yaml")


def test_real_config_has_no_clash() -> None:
    cfg = load_config()
    assert {"bm25", "bm25_tuned", "bm25_champion"} <= set(cfg["retrievers"]) and "sparse" in cfg


def _cfg_with(retrievers: dict) -> dict:
    return {"paths": {}, "retrievers": retrievers}


def test_missing_external_module_has_clear_error() -> None:
    with pytest.raises(ValueError, match="hmr/retrieval/dense.py"):
        build_retriever(_cfg_with({"d": {"type": "dense"}}), "d")


def test_external_builders_and_build_system(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []
    single = ExhaustiveRetriever.__new__(ExhaustiveRetriever)

    class FakeMultiHop:
        def search(self, question: str, k: int) -> RetrievalResult:
            return RetrievalResult("", [], [], 0.0, "mh")

    fake = types.ModuleType("fake_builders")
    fake.build_single = lambda cfg, rcfg: calls.append(("single", rcfg)) or single  # type: ignore[attr-defined]
    fake.build_mh = lambda cfg, rcfg: calls.append(("mh", rcfg)) or FakeMultiHop()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_builders", fake)
    monkeypatch.setitem(factory.EXTERNAL_BUILDERS, "dense", "fake_builders:build_single")
    monkeypatch.setitem(factory.EXTERNAL_BUILDERS, "multihop", "fake_builders:build_mh")
    cfg = _cfg_with({"d": {"type": "dense", "x": 1}, "m": {"type": "multihop_iter"}})

    assert build_retriever(cfg, "d") is single and calls[-1] == ("single", {"type": "dense", "x": 1})
    adapter = build_system(cfg, "d", ["t"])
    assert isinstance(adapter, SingleHopAdapter) and adapter.retriever is single
    mh = build_system(cfg, "m", ["t"])  # type bắt đầu bằng "multihop" → dùng builder multihop, không bọc
    assert isinstance(mh, FakeMultiHop) and calls[-1] == ("mh", {"type": "multihop_iter"})
