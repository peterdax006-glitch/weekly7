from app.solution import anagrams


def test_case_0():
    assert anagrams(*('Listen', 'Silent')) == True

def test_case_1():
    assert anagrams(*('abc', 'abd')) == False

def test_case_2():
    assert anagrams(*('', '')) == True

def test_case_3():
    assert anagrams(*('a gentleman', 'elegant man')) == True
