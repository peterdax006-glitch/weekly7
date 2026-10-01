def median_of(xs):
    """Median of a non-empty list of numbers (mean of the two middle values for even length)."""
    s = sorted(xs)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return (s[m - 1] + s[m]) / 2
