from pathlib import Path
from typing import Any

import pytest

from hmr.data.alignment import AlignmentError, align_examples
from hmr.data.corpus import Corpus
from hmr.data.hotpot import sample_indices


def test_corpus_dedup_by_title(toy_examples: list[dict[str, Any]]) -> None:
    corpus = Corpus.from_examples(toy_examples)
    assert [d["title"] for d in corpus.docs] == ["Film Alpha", "Jean Dupont", "Banana", "Maria Rossi"]
    assert [d["doc_id"] for d in corpus.docs] == [0, 1, 2, 3]
    assert corpus.n_conflicts == 1  # bản NBSP của "Jean Dupont"
    assert corpus.docs[0]["text"] == "Film Alpha is a 1999 drama directed by Jean Dupont. It won two awards."


def test_corpus_save_load(tmp_path: Path, toy_examples: list[dict[str, Any]]) -> None:
    corpus = Corpus.from_examples(toy_examples)
    corpus.save(tmp_path / "c.jsonl")
    loaded = Corpus.load(tmp_path / "c.jsonl")
    assert loaded.docs == corpus.docs and loaded.title_to_id == corpus.title_to_id


def test_alignment_100_percent(toy_examples: list[dict[str, Any]]) -> None:
    corpus = Corpus.from_examples(toy_examples)
    records, report = align_examples([("dev", ex) for ex in toy_examples], corpus)
    assert report.rate == 1.0 and not report.failures
    assert records[0]["gold_doc_ids"] == [0, 1]
    assert records[0]["supporting_facts"][1] == {"title": "Jean Dupont", "sent_id": 1, "doc_id": 1}
    # q2 tham chiếu "Jean Dupont" bản NBSP: vẫn gióng hàng vì so sau chuẩn hoá khoảng trắng
    assert records[1]["gold_doc_ids"] == [1, 3]


def test_alignment_fails_fast_on_missing_title(toy_examples: list[dict[str, Any]]) -> None:
    corpus = Corpus.from_examples(toy_examples[:1])  # thiếu đoạn "Maria Rossi"
    with pytest.raises(AlignmentError):
        align_examples([("dev", ex) for ex in toy_examples], corpus)
    records, report = align_examples([("dev", ex) for ex in toy_examples], corpus, on_error="drop")
    assert [r["qid"] for r in records] == ["q1"] and len(report.failures) == 1


def test_alignment_fails_on_sentence_mismatch(toy_examples: list[dict[str, Any]]) -> None:
    corpus = Corpus.from_examples(toy_examples)
    corpus.docs[1]["sentences"][1] = "Bị dựng sai."
    with pytest.raises(AlignmentError):
        align_examples([("dev", toy_examples[0])], corpus)


def test_source_noise_is_reported_not_failed(toy_examples: list[dict[str, Any]]) -> None:
    ex = toy_examples[0]
    ex["supporting_facts"] = {"title": ["Film Alpha", "Jean Dupont"], "sent_id": [0, 902]}
    corpus = Corpus.from_examples([ex])
    records, report = align_examples([("dev", ex)], corpus)
    assert len(report.source_noise) == 1 and report.rate == 1.0
    assert records[0]["gold_doc_ids"] == [0]


def test_sample_indices_deterministic() -> None:
    a = sample_indices(1000, 50, seed=42)
    assert a == sample_indices(1000, 50, seed=42) and a == sorted(a) and len(set(a)) == 50
    assert sample_indices(10, 50, seed=1) == list(range(10))
