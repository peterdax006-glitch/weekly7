def median(xs):
    s = sorted(xs)
    if not s:
        raise ValueError("median of empty list")
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
