from app.solution import anagrams


def test_case_0():
    assert anagrams(*('Listen', 'Silent')) == True
