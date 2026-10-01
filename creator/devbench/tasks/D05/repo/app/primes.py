def is_prime(k):
    if k < 2:
        return False
    for d in range(2, k):
        if k % d == 0:
            return False
    return True


def primes_below(n):
    return [k for k in range(n) if is_prime(k)]
