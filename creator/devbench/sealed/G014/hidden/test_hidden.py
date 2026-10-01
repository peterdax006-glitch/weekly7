from app.solution import count_vowels


def test_case_0():
    assert count_vowels(*('Hello',)) == 2

def test_case_1():
    assert count_vowels(*('',)) == 0

def test_case_2():
    assert count_vowels(*('AEIOU xyz',)) == 5

def test_case_3():
    assert count_vowels(*('rhythm',)) == 0
