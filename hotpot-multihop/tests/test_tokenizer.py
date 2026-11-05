import hashlib

from hmr.common.config import load_config
from hmr.common.tokenizer import make_analyzer, stopwords, tokenize


def test_lowercase_split_stopwords_stem() -> None:
    assert tokenize("The Runners were RUNNING quickly!") == ["runner", "run", "quickli"]


def test_flags_disable_steps() -> None:
    assert tokenize("The Runners were running", stem=False) == ["runners", "running"]
    assert tokenize("The cats", stem=False, remove_stopwords=False) == ["the", "cats"]


def test_punctuation_and_unicode() -> None:
    assert tokenize("Björk's album (1997), São Paulo", stem=False) == ["björk", "album", "1997", "são", "paulo"]


def test_stopwords_are_nltk_english() -> None:
    sw = stopwords()
    assert {"the", "of", "and", "s"} <= sw
    assert "director" not in sw


def test_analyzer_matches_tokenize() -> None:
    assert make_analyzer(stem=False)("Running dogs") == tokenize("Running dogs", stem=False)


def test_stopwords_match_reference() -> None:
    """Danh sách stopword trong nltk_data không pin version: lệch thì chỉ mục và mọi kết quả đều lệch."""
    digest = hashlib.md5("\n".join(sorted(stopwords())).encode()).hexdigest()
    assert digest == load_config()["data"]["checksums"]["stopwords"], (
        f"Stopword NLTK ({len(stopwords())} từ) khác bản chuẩn 198 từ — tải lại: uv run python -m nltk.downloader stopwords"
    )
