from app.solution import word_freq


def test_case_0():
    assert word_freq(*('the cat the hat',)) == {'the': 2, 'cat': 1, 'hat': 1}
