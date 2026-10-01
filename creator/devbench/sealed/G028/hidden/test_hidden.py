from app.solution import fizzbuzz


def test_case_0():
    assert fizzbuzz(*(15,)) == ['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8', 'Fizz', 'Buzz', '11', 'Fizz', '13', '14', 'FizzBuzz']

def test_case_1():
    assert fizzbuzz(*(1,)) == ['1']

def test_case_2():
    assert fizzbuzz(*(0,)) == []

def test_case_3():
    assert fizzbuzz(*(5,)) == ['1', '2', 'Fizz', '4', 'Buzz']
