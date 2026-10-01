from app.solution import is_palindrome


def test_case_0():
    assert is_palindrome(*('A man, a plan, a canal: Panama',)) == True
