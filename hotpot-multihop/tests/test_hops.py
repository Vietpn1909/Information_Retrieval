from hmr.data.hops import GoldPassage, identify_hop2


def test_title_link() -> None:
    film = GoldPassage(0, "Film Alpha", "Film Alpha is a drama directed by Jean Dupont.")
    person = GoldPassage(1, "Jean Dupont (director)", "Jean Dupont is a director born in France.")
    assert identify_hop2([film, person], "France") == (1, "title_link")
    assert identify_hop2([person, film], "France") == (1, "title_link")


def test_answer_fallback() -> None:
    a = GoldPassage(0, "A", "Mentions B and stuff.")
    b = GoldPassage(1, "B", "Mentions A. Born in Paris.")
    assert identify_hop2([a, b], "Paris") == (1, "answer")


def test_ambiguous() -> None:
    a = GoldPassage(0, "A", "nothing")
    b = GoldPassage(1, "B", "nothing")
    assert identify_hop2([a, b], "Paris") == (None, "ambiguous")
    assert identify_hop2([a], "Paris") == (None, "ambiguous")
