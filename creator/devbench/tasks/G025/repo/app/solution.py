def fizzbuzz(n):
    """Return the FizzBuzz strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' of 5, 'FizzBuzz' of both, else the number."""
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append('FizzBuzz')
        elif i % 3 == 0:
            out.append('Fizz')
        elif i % 5 == 0:
            out.append('Buzz')
        else:
            out.append(str(i))
    return i
