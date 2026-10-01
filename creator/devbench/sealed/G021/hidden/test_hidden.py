from app.solution import is_palindrome


def test_case_0():
    assert is_palindrome(*('A man, a plan, a canal: Panama',)) == True

def test_case_1():
    assert is_palindrome(*('abc',)) == False

def test_case_2():
    assert is_palindrome(*('',)) == True

def test_case_3():
    assert is_palindrome(*('No lemon, no melon',)) == True

def test_case_4():
    assert is_palindrome(*('ab',)) == False
