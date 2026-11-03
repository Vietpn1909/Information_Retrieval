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
