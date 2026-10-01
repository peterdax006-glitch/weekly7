def max_window(xs, k):
    """Largest sum of any k consecutive items; None if k is not between 1 and len(xs)."""
    if k < 1 or k > len(xs):
        return None
    best = cur = sum(xs[:k])
    for i in range(k, len(xs)):
        cur += xs[i] - xs[i - k]
        best = max(best, cur)
    return best
