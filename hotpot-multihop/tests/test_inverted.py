from pathlib import Path

import pytest

from hmr.common.interfaces import Index
from hmr.index.inverted import InvertedIndex


def test_postings_sorted_with_tf(toy_index: InvertedIndex) -> None:
    assert toy_index.postings("apple") == [(0, 2)]
    assert toy_index.postings("banana") == [(0, 1), (1, 1)]
    assert toy_index.postings("cherry") == [(1, 1), (2, 1)]
    assert toy_index.postings("missing") == []


def test_stats(toy_index: Index) -> None:
    assert toy_index.n_docs() == 3
    assert toy_index.doc_len(2) == 4
    assert toy_index.avg_doc_len() == pytest.approx(3.0)
    assert toy_index.df("cherry") == 2 and toy_index.df("missing") == 0
    assert set(toy_index.vocabulary()) == {"apple", "banana", "cherry", "durian", "elder", "fig"}


def test_positions(toy_index: InvertedIndex) -> None:
    assert toy_index.positions("apple", 0) == [0, 2]
    assert toy_index.positions("cherry", 2) == [0]
    assert toy_index.positions("cherry", 0) == []
    assert toy_index.positional_postings("banana") == [(0, [1]), (1, [0])]


def test_doc_ids_must_be_contiguous() -> None:
    index = InvertedIndex()
    index.add_document(0, ["a"])
    with pytest.raises(ValueError):
        index.add_document(2, ["b"])


def test_from_corpus_uses_shared_tokenizer_and_title() -> None:
    docs = [{"doc_id": 0, "title": "Running Man", "text": "The dogs were running."}]
    index = InvertedIndex.from_corpus(docs, stem=True)
    assert index.postings("run") == [(0, 2)]
    assert index.positions("run", 0) == [0, 3]
    assert index.analyze("RUNNING") == ["run"]
    no_title = InvertedIndex.from_corpus(docs, stem=True, include_title=False)
    assert no_title.postings("man") == []


def test_save_load_roundtrip(tmp_path: Path, toy_index: InvertedIndex) -> None:
    toy_index.save(tmp_path / "idx.pkl")
    loaded = InvertedIndex.load(tmp_path / "idx.pkl")
    for term in toy_index.vocabulary():
        assert loaded.postings(term) == toy_index.postings(term)
    assert loaded.stem == toy_index.stem and loaded.n_docs() == 3
