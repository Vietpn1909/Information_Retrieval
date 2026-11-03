from pathlib import Path

from hmr.common.config import DEFAULT_CONFIG_PATH, load_config, retriever_config
from hmr.common.io import read_jsonl, read_qrels, read_trec_run, write_jsonl, write_qrels, write_trec_run


def test_trec_run_roundtrip_and_format(tmp_path: Path) -> None:
    path = tmp_path / "run.txt"
    write_trec_run(path, {"q1": [(5, 2.0), (3, 1.5)], "q2": [(9, 0.1)]}, run_name="bm25")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split() == ["q1", "Q0", "5", "1", "2.000000", "bm25"]
    assert lines[1].split()[3] == "2"
    assert read_trec_run(path) == {"q1": {"5": 2.0, "3": 1.5}, "q2": {"9": 0.1}}


def test_qrels_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "qrels.txt"
    write_qrels(path, {"q1": [4, 7]})
    assert path.read_text(encoding="utf-8") == "q1 0 4 1\nq1 0 7 1\n"
    assert read_qrels(path) == {"q1": {"4": 1, "7": 1}}


def test_jsonl_roundtrip_unicode(tmp_path: Path) -> None:
    recs = [{"title": "Björk", "n": 1}, {"title": "Hà Nội", "n": 2}]
    assert write_jsonl(tmp_path / "x.jsonl", recs) == 2
    assert list(read_jsonl(tmp_path / "x.jsonl")) == recs


def test_default_config_loads_with_absolute_paths() -> None:
    cfg = load_config()
    assert Path(cfg["paths"]["corpus"]).is_absolute()
    assert Path(cfg["paths"]["corpus"]).parent == DEFAULT_CONFIG_PATH.parent.parent / "data"
    bm25 = retriever_config(cfg, "bm25")
    assert bm25["type"] == "bm25" and {"k1", "b", "index"} <= bm25.keys()
