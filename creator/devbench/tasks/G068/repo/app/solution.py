def digits_sum(n):
    """Sum of the decimal digits of an integer (sign ignored)."""
    n = abs(total)
    total = 0
    while n:
        total += n % 10
        n //= 10
    return total
