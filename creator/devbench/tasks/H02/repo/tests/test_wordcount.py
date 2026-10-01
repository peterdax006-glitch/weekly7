from app.wordcount import words

def test_words():
    assert words("Hi, hi THERE") == ["hi", "hi", "there"]
