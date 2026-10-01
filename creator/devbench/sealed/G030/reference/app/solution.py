def gcd(a, b):
    """Greatest common divisor of two non-negative integers (gcd(0, 0) is 0)."""
    while b:
        a, b = b, a % b
    return a
