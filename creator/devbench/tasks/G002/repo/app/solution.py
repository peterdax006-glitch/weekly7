def clamp(x, lo, hi):
    """Return x limited to the closed range [lo, hi]."""
    if hi < lo:
        return lo
    if x > hi:
        return hi
    return x
