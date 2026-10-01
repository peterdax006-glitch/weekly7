from app.solution import title_words


def test_case_0():
    assert title_words(*('hello wORLD',)) == 'Hello World'
